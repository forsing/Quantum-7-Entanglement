# Quantum-7-Entanglement



import csv
import hashlib
import itertools
import random
from pathlib import Path

import numpy as np
import qiskit
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace

random.seed(39)
np.random.seed(39)

FILES = (
    Path("/data/loto7_4716_k89.csv"),
    Path("/data/loto7_4716_k89_loto_2980.csv"),
    Path("/data/loto7_4716_k89_loto_plus_1736.csv"),
)

TITLE = "Quantum Entanglement"

N = 39
K = 7
QUBITS = 6
DIM = 64

WINDOWS = (1, 3)
REGULARIZATIONS = (0.001, 0.01)

TRAIN_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85

BATCH_SIZE = 256
MAX_ITERATIONS = 150
GRADIENT_TOLERANCE = 1e-5

BASIS = np.arange(DIM, dtype=np.int64)

BITS = (
    BASIS[:, None] >> np.arange(QUBITS)[None, :]
) & 1

Z_SIGNS = (1.0 - 2.0 * BITS).T

ZERO_INDICES = tuple(
    np.flatnonzero(BITS[:, qubit] == 0)
    for qubit in range(QUBITS)
)

ONE_INDICES = tuple(
    indices + (1 << qubit)
    for qubit, indices in enumerate(ZERO_INDICES)
)

CNOT_PERMUTATIONS = tuple(
    np.where(
        BITS[:, control] == 1,
        BASIS ^ (1 << (control + 1)),
        BASIS,
    )
    for control in range(QUBITS - 1)
)

# OSNOVNA IDEJA
#
# Svaki registar pocinje u |000000>.
#
# 1. RY i RZ rotacije zavise od prethodnih kola.
# 2. CNOT lanac povezuje kubite:
#    0 -> 1 -> 2 -> 3 -> 4 -> 5.
# 3. Zavrsne RY rotacije menjaju mernu bazu.
#
# Uce se parametri rotacija.
# Zapletenost se nalazi unutar sestokubitnog registra.
# Sedam registara ima zasebne naucene parametre.
#
# Raspodela svakog registra dobija se iz celog
# kompleksnog vektora sa 64 amplitude.
#
# OBUKA
#
# Regresija Bornove raspodele preko Brierovog gubitka.
# Klasicni optimizator menja parametre kvantnih kapija.
# Analiticki adjoint gradijent prolazi kroz celo kolo.
#
# Hronoloska validacija bira prozor istorije i regularizaciju.
# Zavrsna obuka koristi sve raspolozive istorijske prelaze.
#
# NEXT
#
# Logaritmi se racunaju direktno iz amplituda.
# Globalna normalizacija obuhvata sve legalne kombinacije.
# Dinamicko programiranje daje globalni maksimum raspodele.
#
# SIMULACIJA
#
# Batched algebraicki simulator izvodi iste RY, RZ i CX
# operacije kao Qiskit.
# Qiskit Statevector nezavisno proverava amplitude.
# partial_trace proverava redukovane matrice gustine.
#
# Nema uzorkovanja ili slucajnog generisanja kombinacija.



class DistributionError(ArithmeticError):
    pass


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_csv(path):
    raw = path.read_bytes()
    rows = []

    reader = csv.reader(
        raw.decode("utf-8-sig").splitlines()
    )

    for line_number, row in enumerate(reader, start=1):
        try:
            values = tuple(int(value.strip()) for value in row)
        except ValueError as error:
            raise ValueError(
                f"{path.name}: necelobrojan red {line_number}."
            ) from error

        check(
            len(values) == K
            and len(set(values)) == K
            and min(values) >= 1
            and max(values) <= N
            and values == tuple(sorted(values)),
            f"{path.name}: nelegalan red {line_number}.",
        )
        rows.append(values)

    check(
        len(rows) >= 5,
        f"{path.name}: nedovoljno kola za trening, validaciju i test.",
    )

    return {
        "path": path,
        "data": np.asarray(rows, dtype=np.int64) - 1,
        "digest": hashlib.sha256(raw).hexdigest(),
    }


def history_features(data, times, window):
    times = np.asarray(times, dtype=np.int64)

    check(
        window in WINDOWS
        and times.ndim == 1
        and len(times) > 0
        and np.all((times >= 1) & (times <= len(data))),
        "Neispravan prozor ili vreme istorijskog ulaza.",
    )

    features = np.zeros(
        (len(times), 1 + 8 * window),
        dtype=float,
    )
    features[:, 0] = 1.0
    scale = np.sqrt(float(window))

    for lag in range(1, window + 1):
        source = times - lag
        available = source >= 0
        start = 1 + 8 * (lag - 1)

        features[available, start:start + K] = (
            data[source[available]].astype(float)
            + 1.0
            - 20.0
        ) / (19.0 * scale)

        features[available, start + K] = 1.0 / scale

    return features


def initial_parameters(feature_count):
    indices = np.arange(
        QUBITS * feature_count,
        dtype=float,
    ).reshape(QUBITS, feature_count)

    parameters = np.zeros(
        (QUBITS, 2 * feature_count + 1),
        dtype=float,
    )

    parameters[:, :feature_count] = (
        0.04 * np.sin(indices + 1.0)
        / np.sqrt(feature_count)
    )
    parameters[:, feature_count:2 * feature_count] = (
        0.04 * np.cos(indices + 1.0)
        / np.sqrt(feature_count)
    )

    qubits = np.arange(QUBITS, dtype=float) + 1.0

    parameters[:, 0] = (
        np.pi / 2.0 + 0.07 * np.sin(qubits)
    )
    parameters[:, feature_count] = (
        0.50 + 0.08 * np.cos(qubits)
    )
    parameters[:, -1] = (
        0.35 + 0.06 * np.sin(qubits)
    )

    return parameters.ravel()


def unpack_parameters(theta, feature_count):
    theta = np.asarray(theta, dtype=float)

    check(
        theta.shape == (
            QUBITS * (2 * feature_count + 1),
        ),
        "Neispravna dimenzija parametara.",
    )

    matrix = theta.reshape(
        QUBITS,
        2 * feature_count + 1,
    )

    return (
        matrix[:, :feature_count],
        matrix[:, feature_count:2 * feature_count],
        matrix[:, -1],
    )


def circuit_angles(theta, features):
    input_weights, phase_weights, readout_angles = unpack_parameters(
        theta,
        features.shape[1],
    )

    return (
        features @ input_weights.T,
        features @ phase_weights.T,
        readout_angles,
    )


def apply_ry(values, angles, qubit):
    zero = ZERO_INDICES[qubit]
    one = ONE_INDICES[qubit]

    cosine = np.cos(angles / 2.0)[:, None]
    sine = np.sin(angles / 2.0)[:, None]

    result = values.copy()

    result[:, zero] = (
        cosine * values[:, zero]
        - sine * values[:, one]
    )
    result[:, one] = (
        sine * values[:, zero]
        + cosine * values[:, one]
    )

    return result


def apply_rz(values, angles, qubit):
    phase = np.exp(
        -0.5j
        * angles[:, None]
        * Z_SIGNS[qubit][None, :]
    )
    return values * phase


def forward_register(theta, features, record=False):
    input_angles, phase_angles, readout_angles = circuit_angles(
        theta,
        features,
    )

    batch = len(features)
    values = np.zeros(
        (batch, DIM),
        dtype=np.complex128,
    )
    values[:, 0] = 1.0
    cache = []

    for qubit in range(QUBITS):
        angles = input_angles[:, qubit]

        if record:
            cache.append((
                "ry",
                qubit,
                "input",
                angles,
                values,
            ))

        values = apply_ry(values, angles, qubit)
        angles = phase_angles[:, qubit]

        if record:
            cache.append((
                "rz",
                qubit,
                "phase",
                angles,
                values,
            ))

        values = apply_rz(values, angles, qubit)

    for control in range(QUBITS - 1):
        if record:
            cache.append((
                "cx",
                control,
                None,
                None,
                None,
            ))

        values = values[:, CNOT_PERMUTATIONS[control]]

    for qubit in range(QUBITS):
        angles = np.full(
            batch,
            readout_angles[qubit],
            dtype=float,
        )

        if record:
            cache.append((
                "ry",
                qubit,
                "readout",
                angles,
                values,
            ))

        values = apply_ry(values, angles, qubit)

    check(
        np.all(np.isfinite(values))
        and np.allclose(
            np.sum(np.abs(values)**2, axis=1),
            1.0,
            atol=1e-10,
            rtol=0,
        ),
        "Amplitude registra nisu normalizovane.",
    )

    return values, cache


def backward_register(adjoint, cache, features):
    feature_count = features.shape[1]
    gradient = np.zeros(
        (QUBITS, 2 * feature_count + 1),
        dtype=float,
    )

    values = adjoint

    for gate, qubit, group, angles, before in reversed(cache):
        if gate == "cx":
            values = values[:, CNOT_PERMUTATIONS[qubit]]
            continue

        if gate == "ry":
            zero = ZERO_INDICES[qubit]
            one = ONE_INDICES[qubit]

            cosine = np.cos(angles / 2.0)[:, None]
            sine = np.sin(angles / 2.0)[:, None]

            derivative_zero = -0.5 * (
                sine * before[:, zero]
                + cosine * before[:, one]
            )
            derivative_one = 0.5 * (
                cosine * before[:, zero]
                - sine * before[:, one]
            )

            angle_gradient = np.real(np.sum(
                values[:, zero].conj() * derivative_zero
                + values[:, one].conj() * derivative_one,
                axis=1,
            ))

            values = apply_ry(
                values,
                -angles,
                qubit,
            )

        elif gate == "rz":
            after = apply_rz(
                before,
                angles,
                qubit,
            )
            derivative = (
                -0.5j
                * Z_SIGNS[qubit][None, :]
                * after
            )

            angle_gradient = np.real(np.sum(
                values.conj() * derivative,
                axis=1,
            ))

            values = apply_rz(
                values,
                -angles,
                qubit,
            )

        else:
            raise RuntimeError(
                "Nepoznata kapija u povratnom prolazu."
            )

        if group == "input":
            gradient[qubit, :feature_count] += (
                angle_gradient @ features
            )

        elif group == "phase":
            gradient[
                qubit,
                feature_count:2 * feature_count,
            ] += angle_gradient @ features

        elif group == "readout":
            gradient[qubit, -1] += angle_gradient.sum()

        else:
            raise RuntimeError(
                "Nepoznata grupa parametara."
            )

    return gradient.ravel()


def log_amplitudes(amplitudes):
    magnitudes = np.abs(amplitudes)
    result = np.full(magnitudes.shape, -np.inf)

    np.log(
        magnitudes,
        out=result,
        where=magnitudes > 0.0,
    )

    return 2.0 * result


def register_log_probabilities(theta, features):
    amplitudes, _ = forward_register(
        theta,
        features,
    )

    result = log_amplitudes(amplitudes)

    check(
        np.all(np.isfinite(result) | np.isneginf(result))
        and np.allclose(
            logsumexp(result, axis=1),
            0.0,
            atol=1e-10,
            rtol=0,
        ),
        "Logaritamska raspodela registra nije normalizovana.",
    )

    return result


def objective(theta, features, targets, weight):
    total = len(targets)

    check(
        total > 0
        and len(features) == total
        and np.all((targets >= 0) & (targets < N)),
        "Neispravan trening skup.",
    )

    loss = 0.0
    gradient = np.zeros_like(theta)

    for start in range(0, total, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, total)
        batch_features = features[start:stop]
        batch_targets = targets[start:stop]

        amplitudes, cache = forward_register(
            theta,
            batch_features,
            record=True,
        )

        probabilities = np.abs(amplitudes)**2
        difference = probabilities.copy()

        difference[
            np.arange(stop - start),
            batch_targets,
        ] -= 1.0

        loss += float(np.sum(difference**2))

        probability_derivative = 2.0 * difference
        amplitude_adjoint = (
            2.0 * probability_derivative * amplitudes
        )

        gradient += backward_register(
            amplitude_adjoint,
            cache,
            batch_features,
        )

    loss = (
        loss / total
        + weight * np.dot(theta, theta)
    )
    gradient = (
        gradient / total
        + 2.0 * weight * theta
    )

    check(
        np.isfinite(loss)
        and np.all(np.isfinite(gradient)),
        "Nekonacan gubitak ili gradijent.",
    )

    return float(loss), gradient


def fit_models(features, targets, weight, label):
    print(label)
    print("Trening prelaza:", len(targets))
    print("Regularizacija:", weight)

    feature_count = features.shape[1]
    models = []

    for slot in range(K):
        starting_point = initial_parameters(feature_count)
        best_loss = np.inf
        best_theta = starting_point.copy()

        def tracked_objective(candidate):
            nonlocal best_loss, best_theta

            loss, gradient = objective(
                candidate,
                features,
                targets[:, slot],
                weight,
            )

            if loss < best_loss:
                best_loss = loss
                best_theta = candidate.copy()

            return loss, gradient

        print(
            "Obuka registra:",
            slot + 1,
            flush=True,
        )

        result = minimize(
            tracked_objective,
            starting_point,
            method="CG",
            jac=True,
            options={
                "maxiter": MAX_ITERATIONS,
                "gtol": GRADIENT_TOLERANCE,
            },
        )

        chosen = best_theta.copy()
        loss, gradient = objective(
            chosen,
            features,
            targets[:, slot],
            weight,
        )
        gradient_norm = float(
            np.max(np.abs(gradient))
        )

        check(
            np.isfinite(best_loss)
            and np.all(np.isfinite(chosen)),
            "Optimizator nije sacuvao konacne parametre.",
        )

        print("Status optimizatora:", result.message)
        print("Sacuvani gubitak:", f"{loss:.8f}")
        print(
            "Norma gradijenta sacuvane tacke:",
            f"{gradient_norm:.8g}",
        )

        if (
            not result.success
            or gradient_norm > GRADIENT_TOLERANCE
        ):
            print(
                "UPOZORENJE: konvergencija nije potvrdjena."
            )

        models.append(chosen)

    return models


def predict_logs(models, features):
    check(
        len(models) == K,
        "Nedostaju modeli registara.",
    )

    result = np.empty(
        (len(features), K, DIM),
        dtype=float,
    )

    for slot, theta in enumerate(models):
        for start in range(0, len(features), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(features))

            result[start:stop, slot] = (
                register_log_probabilities(
                    theta,
                    features[start:stop],
                )
            )

    return result


def joint_distribution(log_weights):
    logs = np.asarray(log_weights, dtype=float).copy()

    check(
        logs.ndim == 2
        and np.all(np.isfinite(logs) | np.isneginf(logs)),
        "Neispravni logaritmi zajednicke raspodele.",
    )

    slots, states = logs.shape
    check(
        1 <= slots <= states,
        "Neispravne dimenzije zajednicke raspodele.",
    )

    indices = np.arange(states)
    increasing = indices[:, None] < indices[None, :]

    allowed = (
        (indices[None, :] >= np.arange(slots)[:, None])
        & (
            indices[None, :]
            <= states - slots + np.arange(slots)[:, None]
        )
    )
    logs[~allowed] = -np.inf

    forward = np.full_like(logs, -np.inf)
    backward = np.full_like(logs, -np.inf)
    forward[0] = logs[0]

    maximum = logs[0].copy()
    parents = np.zeros(
        (slots, states),
        dtype=np.int64,
    )

    for slot in range(1, slots):
        forward[slot] = logs[slot] + logsumexp(
            np.where(
                increasing,
                forward[slot - 1, :, None],
                -np.inf,
            ),
            axis=0,
        )

        candidates = np.where(
            increasing,
            maximum[:, None],
            -np.inf,
        )
        parents[slot] = candidates.argmax(axis=0)
        maximum = logs[slot] + candidates[
            parents[slot],
            np.arange(states),
        ]

    log_partition = float(
        logsumexp(forward[-1])
    )

    if not np.isfinite(log_partition):
        raise DistributionError(
            "Kvantno stanje nema pozitivnu legalnu masu."
        )

    path = np.zeros(slots, dtype=np.int64)
    path[-1] = int(maximum.argmax())

    for slot in range(slots - 1, 0, -1):
        path[slot - 1] = parents[slot, path[slot]]

    backward[-1] = 0.0

    for slot in range(slots - 2, -1, -1):
        backward[slot] = logsumexp(
            np.where(
                increasing,
                (
                    logs[slot + 1]
                    + backward[slot + 1]
                )[None, :],
                -np.inf,
            ),
            axis=1,
        )

    marginals = np.exp(
        forward + backward - log_partition
    )
    mode_log_mass = float(
        logs[np.arange(slots), path].sum()
    )

    check(
        np.all(np.diff(path) > 0)
        and np.isfinite(mode_log_mass)
        and mode_log_mass <= log_partition + 1e-8
        and log_partition <= 1e-8,
        "Neispravan maksimum zajednicke raspodele.",
    )
    check(
        np.allclose(
            marginals.sum(axis=1),
            1.0,
            atol=1e-8,
            rtol=0,
        ),
        "Zajednicka raspodela nije normalizovana.",
    )

    return {
        "path": path,
        "marginals": marginals,
        "log_partition": log_partition,
        "log_mode_probability": (
            mode_log_mass - log_partition
        ),
        "logs": logs,
    }


def evaluate(models, features, targets):
    check(
        len(features) == len(targets)
        and len(targets) > 0,
        "Neispravan evaluacioni skup.",
    )

    predicted_logs = predict_logs(models, features)
    losses = []
    hits = []

    for row, target in enumerate(targets):
        distribution = joint_distribution(
            predicted_logs[row, :, :N]
        )

        target_log_mass = float(
            distribution["logs"][
                np.arange(K), target
            ].sum()
        )

        loss = (
            distribution["log_partition"]
            - target_log_mass
        )

        check(
            not np.isnan(loss) and loss >= -1e-8,
            "Neispravan evaluacioni NLL.",
        )

        losses.append(loss)
        hits.append(
            len(set(distribution["path"]) & set(target))
        )

    losses = np.asarray(losses, dtype=float)

    return {
        "nll": float(losses.mean()),
        "hits": np.asarray(hits, dtype=np.int64),
        "zero_targets": int(
            np.isposinf(losses).sum()
        ),
    }


def single_qubit_purities(state):
    state = np.asarray(state, dtype=complex)

    check(
        state.shape == (DIM,)
        and np.isclose(
            np.vdot(state, state).real,
            1.0,
            atol=1e-10,
            rtol=0,
        ),
        "Neispravno stanje za proveru cistoce.",
    )

    result = np.empty(QUBITS, dtype=float)

    for qubit in range(QUBITS):
        zero = state[ZERO_INDICES[qubit]]
        one = state[ONE_INDICES[qubit]]

        probability_zero = float(
            np.vdot(zero, zero).real
        )
        probability_one = float(
            np.vdot(one, one).real
        )
        coherence = np.vdot(one, zero)

        result[qubit] = (
            probability_zero**2
            + probability_one**2
            + 2.0 * np.abs(coherence)**2
        )

    check(
        np.all(result >= 0.5 - 1e-9)
        and np.all(result <= 1.0 + 1e-9),
        "Neispravne cistoce redukovanih stanja.",
    )

    return result


def verify_qiskit(theta, features):
    expected, _ = forward_register(
        theta,
        features,
    )
    input_angles, phase_angles, readout_angles = circuit_angles(
        theta,
        features,
    )
    predicted_logs = register_log_probabilities(
        theta,
        features,
    )

    purities = np.empty(
        (len(features), QUBITS),
        dtype=float,
    )

    for row in range(len(features)):
        circuit = QuantumCircuit(QUBITS)

        for qubit in range(QUBITS):
            circuit.ry(
                float(input_angles[row, qubit]),
                qubit,
            )
            circuit.rz(
                float(phase_angles[row, qubit]),
                qubit,
            )

        for control in range(QUBITS - 1):
            circuit.cx(control, control + 1)

        for qubit in range(QUBITS):
            circuit.ry(
                float(readout_angles[qubit]),
                qubit,
            )

        state = Statevector.from_instruction(circuit)
        actual = np.asarray(state.data)

        check(
            sum(
                item.operation.name == "cx"
                for item in circuit.data
            ) == QUBITS - 1,
            "Neispravan broj CNOT kapija.",
        )

        check(
            np.allclose(
                actual,
                expected[row],
                atol=2e-11,
                rtol=2e-11,
            ),
            "Qiskit i algebraicke amplitude nisu usaglasene.",
        )
        check(
            np.allclose(
                np.abs(actual)**2,
                np.exp(predicted_logs[row]),
                atol=2e-11,
                rtol=0,
            ),
            "Qiskit i logaritamske verovatnoce nisu usaglasene.",
        )

        purities[row] = single_qubit_purities(actual)

        for qubit in range(QUBITS):
            traced_qubits = [
                other
                for other in range(QUBITS)
                if other != qubit
            ]
            reduced = partial_trace(
                state,
                traced_qubits,
            )
            qiskit_purity = float(np.real(np.trace(
                reduced.data @ reduced.data
            )))

            check(
                np.isclose(
                    purities[row, qubit],
                    qiskit_purity,
                    atol=2e-10,
                    rtol=0,
                ),
                "Provere redukovane matrice gustine nisu usaglasene.",
            )

    return purities


def self_checks(data):
    # Bell stanje proverava CNOT i little-endian redosled.
    bell = np.zeros((1, DIM), dtype=complex)
    bell[:, 0] = 1.0
    bell = apply_ry(
        bell,
        np.array([np.pi / 2.0]),
        0,
    )
    bell = bell[:, CNOT_PERMUTATIONS[0]]

    bell_circuit = QuantumCircuit(QUBITS)
    bell_circuit.h(0)
    bell_circuit.cx(0, 1)

    qiskit_bell = Statevector.from_instruction(
        bell_circuit
    ).data
    bell_purities = single_qubit_purities(bell[0])

    check(
        np.allclose(
            bell[0],
            qiskit_bell,
            atol=1e-12,
            rtol=0,
        )
        and np.allclose(
            bell_purities[:2],
            0.5,
            atol=1e-12,
            rtol=0,
        )
        and np.allclose(
            bell_purities[2:],
            1.0,
            atol=1e-12,
            rtol=0,
        ),
        "Provera Bell stanja nije prosla.",
    )

    sample_times = np.arange(
        1,
        min(len(data), 6),
    )
    sample_targets = data[sample_times, 0]

    for window in WINDOWS:
        features = history_features(
            data,
            sample_times,
            window,
        )
        theta = initial_parameters(features.shape[1])
        theta += 0.02 * np.sin(
            np.arange(len(theta)) + 1.0
        )

        verify_qiskit(theta, features[:2])

        _, analytic = objective(
            theta,
            features,
            sample_targets,
            0.001,
        )

        for index in range(len(theta)):
            offset = np.zeros_like(theta)
            offset[index] = 1e-6

            numerical = (
                objective(
                    theta + offset,
                    features,
                    sample_targets,
                    0.001,
                )[0]
                - objective(
                    theta - offset,
                    features,
                    sample_targets,
                    0.001,
                )[0]
            ) / 2e-6

            check(
                np.isclose(
                    numerical,
                    analytic[index],
                    atol=2e-6,
                    rtol=2e-4,
                ),
                f"Gradijent nije potvrdjen za prozor "
                f"{window}, parametar {index}.",
            )

        print(
            "Provere kola i gradijenta su prosle za prozor:",
            window,
            flush=True,
        )

    # Kontrola fazne interferencije GHZ stanja.
    # Paritet se koristi iskljucivo u ovom matematickom testu.
    feature_count = 1 + 8 * WINDOWS[0]
    features = np.zeros((1, feature_count))
    features[:, 0] = 1.0

    phase_parameters = np.zeros(
        (QUBITS, 2 * feature_count + 1)
    )
    phase_parameters[0, 0] = np.pi / 2.0
    phase_parameters[:, -1] = np.pi / 2.0

    first_state, _ = forward_register(
        phase_parameters.ravel(),
        features,
    )

    phase_parameters[0, feature_count] = np.pi

    second_state, _ = forward_register(
        phase_parameters.ravel(),
        features,
    )

    even = (BITS.sum(axis=1) % 2) == 0

    check(
        np.sum(np.abs(first_state[0, even])**2)
        > 1.0 - 1e-10
        and np.sum(np.abs(second_state[0, ~even])**2)
        > 1.0 - 1e-10
        and np.allclose(
            single_qubit_purities(first_state[0]),
            0.5,
            atol=1e-10,
            rtol=0,
        ),
        "Provera GHZ fazne interferencije nije prosla.",
    )

    verify_qiskit(
        phase_parameters.ravel(),
        features,
    )

    zero_parameters = np.zeros(
        QUBITS * (2 * feature_count + 1)
    )
    zero_logs = register_log_probabilities(
        zero_parameters,
        features,
    )

    check(
        np.isclose(zero_logs[0, 0], 0.0)
        and np.all(np.isneginf(zero_logs[0, 1:])),
        "Provera baznog stanja nije prosla.",
    )

    tiny_log = log_amplitudes(
        np.array([1e-180 + 0j])
    )[0]

    check(
        np.isfinite(tiny_log)
        and np.isclose(
            tiny_log,
            2.0 * np.log(1e-180),
            atol=1e-10,
            rtol=0,
        ),
        "Provera male amplitude nije prosla.",
    )

    toy = softmax(
        np.sin(np.arange(15).reshape(3, 5)),
        axis=1,
    )
    distribution = joint_distribution(np.log(toy))

    scores = [
        sum(
            np.log(toy[slot, combination[slot]])
            for slot in range(3)
        )
        for combination in itertools.combinations(range(5), 3)
    ]

    path = distribution["path"]
    mode_score = sum(
        np.log(toy[slot, path[slot]])
        for slot in range(3)
    )

    check(
        np.isclose(
            distribution["log_partition"],
            logsumexp(scores),
            atol=1e-10,
            rtol=0,
        )
        and np.isclose(
            mode_score,
            max(scores),
            atol=1e-10,
            rtol=0,
        ),
        "Iscrpna provera normalizacije i maksimuma nije prosla.",
    )

    single = np.full((3, 3), -np.inf)
    np.fill_diagonal(single, 0.0)
    single_result = joint_distribution(single)

    check(
        np.array_equal(
            single_result["path"],
            np.arange(3),
        )
        and np.isclose(
            single_result["log_partition"],
            0.0,
            atol=1e-12,
            rtol=0,
        ),
        "Provera jedinog legalnog stanja nije prosla.",
    )

    try:
        joint_distribution(
            np.full((3, 3), -np.inf)
        )
    except DistributionError:
        pass
    else:
        raise RuntimeError(
            "Nulta legalna masa nije pravilno odbijena."
        )

    print("Sve ugradjene pocetne provere su prosle.")


def print_metrics(label, metrics):
    print(label)
    print("Broj kola:", len(metrics["hits"]))
    print("Srednji NLL:", f'{metrics["nll"]:.8f}')
    print(
        "Srednji broj pogodaka:",
        f'{metrics["hits"].mean():.6f}',
    )
    print(
        "Nulte ciljne verovatnoce:",
        metrics["zero_targets"],
    )
    print(
        "Broj kola po pogocima od 0 do 7:",
        np.bincount(
            metrics["hits"],
            minlength=K + 1,
        ).tolist(),
    )


def process_dataset(dataset):
    path = dataset["path"]
    data = dataset["data"]

    targets = data[1:]
    times = np.arange(1, len(data))
    total = len(targets)

    train_end = int(TRAIN_FRACTION * total)
    validation_end = int(
        VALIDATION_END_FRACTION * total
    )

    check(
        0 < train_end < validation_end < total,
        "Neispravna hronoloska podela.",
    )

    print()
    print("CSV:", path.name)
    print("Redova:", len(data))
    print("SHA256:", dataset["digest"])
    print("Trening prelaza:", train_end)
    print(
        "Validacionih prelaza:",
        validation_end - train_end,
    )
    print("Test prelaza:", total - validation_end)

    best_settings = None
    best_validation = np.inf

    for window in WINDOWS:
        features = history_features(
            data,
            times,
            window,
        )

        for weight in REGULARIZATIONS:
            print()
            print(
                "Kandidat:",
                "prozor",
                window,
                "regularizacija",
                weight,
            )

            models = fit_models(
                features[:train_end],
                targets[:train_end],
                weight,
                "UCENJE ZA VALIDACIJU",
            )

            try:
                metrics = evaluate(
                    models,
                    features[train_end:validation_end],
                    targets[train_end:validation_end],
                )
            except DistributionError as error:
                print(
                    "Kandidat nema definisanu legalnu raspodelu."
                )
                print(str(error))
                continue

            print_metrics(
                "VALIDACIONI REZULTAT",
                metrics,
            )

            if (
                np.isfinite(metrics["nll"])
                and metrics["nll"] < best_validation
            ):
                best_validation = metrics["nll"]
                best_settings = (window, weight)

    check(
        best_settings is not None,
        "Nijedan kandidat nema konacan validacioni NLL.",
    )

    window, weight = best_settings
    print()
    print("Izabrani prozor:", window)
    print("Izabrana regularizacija:", weight)

    features = history_features(
        data,
        times,
        window,
    )

    test_models = fit_models(
        features[:validation_end],
        targets[:validation_end],
        weight,
        "UCENJE ZA ODVOJENI TEST",
    )

    test_metrics = evaluate(
        test_models,
        features[validation_end:],
        targets[validation_end:],
    )

    print_metrics(
        "ODVOJENI TEST",
        test_metrics,
    )

    # Parametri su zamrznuti tokom testa.
    # Ranija test kola mogu biti istorijski ulaz kasnijeg kola.
    # Test rezultat ne bira hiperparametre.

    final_models = fit_models(
        features,
        targets,
        weight,
        "ZAVRSNO UCENJE NA SVIM ISTORIJSKIM PRELAZIMA",
    )

    next_features = history_features(
        data,
        np.array([len(data)]),
        window,
    )

    print("Cistoce pojedinacnih kubita u NEXT stanjima:")

    for slot, theta in enumerate(final_models):
        purities = verify_qiskit(
            theta,
            next_features,
        )[0]

        print(
            "Registar:",
            slot + 1,
            "cistoce:",
            " ".join(
                f"{value:.8f}"
                for value in purities
            ),
        )
        print(
            "Kubita sa cistocom ispod 1 - 1e-8:",
            int(np.sum(purities < 1.0 - 1e-8)),
        )

    print("Zavrsne Qiskit provere svih sedam registara su prosle.")

    next_logs = predict_logs(
        final_models,
        next_features,
    )[0]

    distribution = joint_distribution(
        next_logs[:, :N]
    )

    prediction = distribution["path"]
    combination = prediction + 1

    print(
        "Log legalne mase:",
        f'{distribution["log_partition"]:.8f}',
    )
    print(
        "Log modelske verovatnoce NEXT:",
        f'{distribution["log_mode_probability"]:.8f}',
    )
    print(
        "Najvece marginalne verovatnoce:",
        " ".join(
            f"{value:.5f}"
            for value in distribution["marginals"].max(axis=1)
        ),
    )

    probe_times = np.unique(
        np.array([
            min(window, len(data)),
            len(data) // 2,
            len(data),
        ], dtype=int)
    )
    probe_features = history_features(
        data,
        probe_times,
        window,
    )
    probe_probabilities = np.exp(
        predict_logs(final_models, probe_features)
    )

    variation = float(np.max(np.abs(
        probe_probabilities
        - np.exp(next_logs)[None, :, :]
    )))

    print(
        "Promena raspodele izmedju stvarnih istorijskih konteksta:",
        f"{variation:.10g}",
    )

    if np.array_equal(prediction, data[-1]):
        print("PROVERA: NEXT ponavlja poslednje kolo.")

    longest_run = 1
    current_run = 1

    for difference in np.diff(combination):
        if difference == 1:
            current_run += 1
            longest_run = max(
                longest_run,
                current_run,
            )
        else:
            current_run = 1

    if longest_run >= 5:
        print(
            "PROVERA: najmanje pet uzastopnih brojeva."
        )

    span = int(combination[-1] - combination[0])
    historical_spans = data[:, -1] - data[:, 0]
    span_percentile = float(
        100.0 * np.mean(historical_spans <= span)
    )

    print("Raspon NEXT kombinacije:", span)
    print(
        "Udeo istorijskih raspona manjih ili jednakih NEXT rasponu:",
        f"{span_percentile:.3f}%",
    )

    return {
        "name": path.name,
        "digest": dataset["digest"],
        "next": combination,
        "probabilities": np.exp(next_logs),
        "marginals": distribution["marginals"],
    }


def main():
    print(TITLE)
    print("Qiskit:", qiskit.__version__)
    print("Sedam registara sa po sest kubita.")
    print("Rotacije zavisne od istorije i CNOT lanac.")
    print("Provera zapletenosti preko redukovanih matrica gustine.")
    print("Normalizacija i maksimum svih legalnih kombinacija.")
    print("NLL je negativni logaritam modelske verovatnoce stvarnog kola.")

    datasets = [
        load_csv(path)
        for path in FILES
    ]

    self_checks(datasets[0]["data"])

    results = [
        process_dataset(dataset)
        for dataset in datasets
    ]

    for first, second in itertools.combinations(results, 2):
        if first["digest"] == second["digest"]:
            print("PROVERA: dva ulaza imaju isti SHA256.")
            print(first["name"])
            print(second["name"])

        if np.array_equal(
            first["next"],
            second["next"],
        ):
            print()
            print("PROVERA ISTIH NEXT KOMBINACIJA")
            print(first["name"])
            print(second["name"])
            print(
                "Najveca razlika sirovih raspodela:",
                float(np.max(np.abs(
                    first["probabilities"]
                    - second["probabilities"]
                ))),
            )
            print(
                "Najveca razlika legalnih marginalnih raspodela:",
                float(np.max(np.abs(
                    first["marginals"]
                    - second["marginals"]
                ))),
            )

    print()
    print("ZAVRSNE NEXT KOMBINACIJE")

    for result in results:
        print(result["name"])
        print(
            "NEXT:",
            " ".join(map(str, result["next"])),
        )


if __name__ == "__main__":
    main()



"""
Quantum Entanglement
Qiskit: 1.4.4
Sedam registara sa po sest kubita.
Rotacije zavisne od istorije i CNOT lanac.
Provera zapletenosti preko redukovanih matrica gustine.
Normalizacija i maksimum svih legalnih kombinacija.
NLL je negativni logaritam modelske verovatnoce stvarnog kola.
Provere kola i gradijenta su prosle za prozor: 1
Provere kola i gradijenta su prosle za prozor: 3
Sve ugradjene pocetne provere su prosle.

CSV: loto7_4716_k89.csv
Redova: 4716
SHA256: deb02ea741f500e3270402730ed89565f1c9b4618d53e5a08fea2b7c2c2af31a
Trening prelaza: 3300
Validacionih prelaza: 707
Test prelaza: 708

Kandidat: prozor 1 regularizacija 0.001
UCENJE ZA VALIDACIJU
Trening prelaza: 3300
Regularizacija: 0.001
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.90657413
Norma gradijenta sacuvane tacke: 9.6241908e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95295621
Norma gradijenta sacuvane tacke: 7.6288144e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96290654
Norma gradijenta sacuvane tacke: 9.307891e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96440423
Norma gradijenta sacuvane tacke: 7.6139334e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96177362
Norma gradijenta sacuvane tacke: 9.7952567e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95119985
Norma gradijenta sacuvane tacke: 8.1254455e-06
Obuka registra: 7
Status optimizatora: Maximum number of iterations has been exceeded.
Sacuvani gubitak: 0.91039377
Norma gradijenta sacuvane tacke: 5.8651643e-05
UPOZORENJE: konvergencija nije potvrdjena.
VALIDACIONI REZULTAT
Broj kola: 707
Srednji NLL: 19.07770006
Srednji broj pogodaka: 1.236209
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [152, 305, 193, 46, 10, 1, 0, 0]

Kandidat: prozor 1 regularizacija 0.01
UCENJE ZA VALIDACIJU
Trening prelaza: 3300
Regularizacija: 0.01
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95426368
Norma gradijenta sacuvane tacke: 4.639433e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99610309
Norma gradijenta sacuvane tacke: 7.2854236e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00419766
Norma gradijenta sacuvane tacke: 5.789212e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00716187
Norma gradijenta sacuvane tacke: 6.0112589e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00476423
Norma gradijenta sacuvane tacke: 8.6679158e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98648190
Norma gradijenta sacuvane tacke: 6.9193184e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96771003
Norma gradijenta sacuvane tacke: 9.9770701e-06
VALIDACIONI REZULTAT
Broj kola: 707
Srednji NLL: 18.73034262
Srednji broj pogodaka: 1.248939
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [149, 308, 184, 58, 7, 1, 0, 0]

Kandidat: prozor 3 regularizacija 0.001
UCENJE ZA VALIDACIJU
Trening prelaza: 3300
Regularizacija: 0.001
Obuka registra: 1
Status optimizatora: Maximum number of iterations has been exceeded.
Sacuvani gubitak: 0.90577779
Norma gradijenta sacuvane tacke: 1.5145271e-05
UPOZORENJE: konvergencija nije potvrdjena.
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95189338
Norma gradijenta sacuvane tacke: 5.3948247e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96205159
Norma gradijenta sacuvane tacke: 6.7348315e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96382812
Norma gradijenta sacuvane tacke: 4.5992223e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96103489
Norma gradijenta sacuvane tacke: 3.9296994e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95023021
Norma gradijenta sacuvane tacke: 9.3486641e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.90856153
Norma gradijenta sacuvane tacke: 8.1177034e-06
VALIDACIONI REZULTAT
Broj kola: 707
Srednji NLL: 19.02315291
Srednji broj pogodaka: 1.230552
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [156, 308, 177, 57, 8, 1, 0, 0]

Kandidat: prozor 3 regularizacija 0.01
UCENJE ZA VALIDACIJU
Trening prelaza: 3300
Regularizacija: 0.01
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95198324
Norma gradijenta sacuvane tacke: 6.393852e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99371479
Norma gradijenta sacuvane tacke: 4.9716174e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00272458
Norma gradijenta sacuvane tacke: 4.0315149e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00580510
Norma gradijenta sacuvane tacke: 9.0115143e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00345873
Norma gradijenta sacuvane tacke: 4.4256769e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98410409
Norma gradijenta sacuvane tacke: 7.2142333e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96464888
Norma gradijenta sacuvane tacke: 5.0638333e-06
VALIDACIONI REZULTAT
Broj kola: 707
Srednji NLL: 18.61470226
Srednji broj pogodaka: 1.248939
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [153, 303, 186, 53, 11, 1, 0, 0]

Izabrani prozor: 3
Izabrana regularizacija: 0.01
UCENJE ZA ODVOJENI TEST
Trening prelaza: 4007
Regularizacija: 0.01
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95219379
Norma gradijenta sacuvane tacke: 9.9132267e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99389610
Norma gradijenta sacuvane tacke: 5.644879e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00284296
Norma gradijenta sacuvane tacke: 8.8740816e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00586326
Norma gradijenta sacuvane tacke: 5.269017e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00338738
Norma gradijenta sacuvane tacke: 8.2369743e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98386340
Norma gradijenta sacuvane tacke: 9.8032995e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96441576
Norma gradijenta sacuvane tacke: 8.3197322e-06
ODVOJENI TEST
Broj kola: 708
Srednji NLL: 18.56537770
Srednji broj pogodaka: 1.223164
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [163, 294, 185, 62, 4, 0, 0, 0]
ZAVRSNO UCENJE NA SVIM ISTORIJSKIM PRELAZIMA
Trening prelaza: 4715
Regularizacija: 0.01
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95221837
Norma gradijenta sacuvane tacke: 9.0625579e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99382900
Norma gradijenta sacuvane tacke: 5.7938888e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00307056
Norma gradijenta sacuvane tacke: 6.9039696e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00601990
Norma gradijenta sacuvane tacke: 4.5086708e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00315817
Norma gradijenta sacuvane tacke: 9.2354353e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98352419
Norma gradijenta sacuvane tacke: 7.3150273e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96451130
Norma gradijenta sacuvane tacke: 8.6222621e-06
Cistoce pojedinacnih kubita u NEXT stanjima:
Registar: 1 cistoce: 0.99489836 0.98093458 0.92063199 0.77488306 0.69502675 0.80709909
Kubita sa cistocom ispod 1 - 1e-8: 6
Registar: 2 cistoce: 0.99004900 0.98531089 0.97362672 0.96049283 0.84151239 0.85468235
Kubita sa cistocom ispod 1 - 1e-8: 6
Registar: 3 cistoce: 0.98786921 0.97706109 0.97314152 0.91309120 0.87688636 0.94145474
Kubita sa cistocom ispod 1 - 1e-8: 6
Registar: 4 cistoce: 0.98799898 0.97324484 0.97560204 0.95602830 0.95783145 0.99237083
Kubita sa cistocom ispod 1 - 1e-8: 6
Registar: 5 cistoce: 0.98795771 0.98023748 0.99037441 0.95394459 0.93344710 0.97576031
Kubita sa cistocom ispod 1 - 1e-8: 6
Registar: 6 cistoce: 0.99496322 0.98680068 0.87664069 0.55650482 0.54559780 0.80903189
Kubita sa cistocom ispod 1 - 1e-8: 6
Registar: 7 cistoce: 0.97443706 0.97351457 0.99828816 0.70332092 0.64005566 0.84391016
Kubita sa cistocom ispod 1 - 1e-8: 6
Zavrsne Qiskit provere svih sedam registara su prosle.
Log legalne mase: -5.32495099
Log modelske verovatnoce NEXT: -12.02844206
Najvece marginalne verovatnoce: 0.35492 0.14513 0.12784 0.13356 0.13344 0.17912 0.30696
Promena raspodele izmedju stvarnih istorijskih konteksta: 0.0276639502
Raspon NEXT kombinacije: 38
Udeo istorijskih raspona manjih ili jednakih NEXT rasponu: 100.000%

CSV: loto7_4716_k89_loto_2980.csv
Redova: 2979
SHA256: f1cd81cd04660d50d9058675924da34346fc006ce8dabd84d52266c66afe17d1
Trening prelaza: 2084
Validacionih prelaza: 447
Test prelaza: 447

Kandidat: prozor 1 regularizacija 0.001
UCENJE ZA VALIDACIJU
Trening prelaza: 2084
Regularizacija: 0.001
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.90713319
Norma gradijenta sacuvane tacke: 7.7019667e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95252036
Norma gradijenta sacuvane tacke: 8.2476634e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96277193
Norma gradijenta sacuvane tacke: 9.3700733e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96464160
Norma gradijenta sacuvane tacke: 8.7536821e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96123804
Norma gradijenta sacuvane tacke: 6.1729684e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95099168
Norma gradijenta sacuvane tacke: 8.2984643e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.90960386
Norma gradijenta sacuvane tacke: 9.8415446e-06
VALIDACIONI REZULTAT
Broj kola: 447
Srednji NLL: 19.10907634
Srednji broj pogodaka: 1.217002
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [97, 207, 101, 33, 9, 0, 0, 0]

Kandidat: prozor 1 regularizacija 0.01
UCENJE ZA VALIDACIJU
Trening prelaza: 2084
Regularizacija: 0.01
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95497416
Norma gradijenta sacuvane tacke: 7.7195157e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99549421
Norma gradijenta sacuvane tacke: 4.6856222e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00389753
Norma gradijenta sacuvane tacke: 7.2416378e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00747111
Norma gradijenta sacuvane tacke: 8.2220196e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00418954
Norma gradijenta sacuvane tacke: 7.635673e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98652716
Norma gradijenta sacuvane tacke: 8.3355221e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96717249
Norma gradijenta sacuvane tacke: 7.538698e-06
VALIDACIONI REZULTAT
Broj kola: 447
Srednji NLL: 18.88202400
Srednji broj pogodaka: 1.230425
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [99, 192, 116, 35, 4, 1, 0, 0]

Kandidat: prozor 3 regularizacija 0.001
UCENJE ZA VALIDACIJU
Trening prelaza: 2084
Regularizacija: 0.001
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.90591107
Norma gradijenta sacuvane tacke: 8.597111e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95091777
Norma gradijenta sacuvane tacke: 6.8572737e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96181892
Norma gradijenta sacuvane tacke: 7.602055e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96361035
Norma gradijenta sacuvane tacke: 4.5610042e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96024642
Norma gradijenta sacuvane tacke: 9.673648e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95045741
Norma gradijenta sacuvane tacke: 6.7602885e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.90728597
Norma gradijenta sacuvane tacke: 5.6343299e-06
VALIDACIONI REZULTAT
Broj kola: 447
Srednji NLL: 19.12894142
Srednji broj pogodaka: 1.203579
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [93, 206, 116, 28, 4, 0, 0, 0]

Kandidat: prozor 3 regularizacija 0.01
UCENJE ZA VALIDACIJU
Trening prelaza: 2084
Regularizacija: 0.01
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95241373
Norma gradijenta sacuvane tacke: 8.0643283e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99315299
Norma gradijenta sacuvane tacke: 8.7338954e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00231191
Norma gradijenta sacuvane tacke: 8.0587362e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00603944
Norma gradijenta sacuvane tacke: 6.7480916e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00295134
Norma gradijenta sacuvane tacke: 9.6237592e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98468022
Norma gradijenta sacuvane tacke: 7.8224118e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96418573
Norma gradijenta sacuvane tacke: 7.3157317e-06
VALIDACIONI REZULTAT
Broj kola: 447
Srednji NLL: 18.75171777
Srednji broj pogodaka: 1.201342
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [101, 200, 110, 29, 5, 2, 0, 0]

Izabrani prozor: 3
Izabrana regularizacija: 0.01
UCENJE ZA ODVOJENI TEST
Trening prelaza: 2531
Regularizacija: 0.01
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95320259
Norma gradijenta sacuvane tacke: 7.2940611e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99381659
Norma gradijenta sacuvane tacke: 5.449964e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00247369
Norma gradijenta sacuvane tacke: 8.1351165e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00592253
Norma gradijenta sacuvane tacke: 5.4412494e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00303629
Norma gradijenta sacuvane tacke: 5.8227229e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98402373
Norma gradijenta sacuvane tacke: 9.3999242e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96474675
Norma gradijenta sacuvane tacke: 9.6164221e-06
ODVOJENI TEST
Broj kola: 447
Srednji NLL: 18.73407482
Srednji broj pogodaka: 1.261745
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [95, 193, 113, 40, 5, 1, 0, 0]
ZAVRSNO UCENJE NA SVIM ISTORIJSKIM PRELAZIMA
Trening prelaza: 2978
Regularizacija: 0.01
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95339321
Norma gradijenta sacuvane tacke: 5.4947208e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99382136
Norma gradijenta sacuvane tacke: 8.0288403e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00294709
Norma gradijenta sacuvane tacke: 5.5596014e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00631027
Norma gradijenta sacuvane tacke: 9.8344171e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00288077
Norma gradijenta sacuvane tacke: 9.5423977e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98375020
Norma gradijenta sacuvane tacke: 9.317054e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96437416
Norma gradijenta sacuvane tacke: 4.9012863e-06
Cistoce pojedinacnih kubita u NEXT stanjima:
Registar: 1 cistoce: 0.99303522 0.97825575 0.91679049 0.76905195 0.68719699 0.79894990
Kubita sa cistocom ispod 1 - 1e-8: 6
Registar: 2 cistoce: 0.98573707 0.97721590 0.97889709 0.96268743 0.82682382 0.84431509
Kubita sa cistocom ispod 1 - 1e-8: 6
Registar: 3 cistoce: 0.98263860 0.96214928 0.95348186 0.91726891 0.88046726 0.93182385
Kubita sa cistocom ispod 1 - 1e-8: 6
Registar: 4 cistoce: 0.98276167 0.96862972 0.96886096 0.95623344 0.96084400 0.98788530
Kubita sa cistocom ispod 1 - 1e-8: 6
Registar: 5 cistoce: 0.98336339 0.97064361 0.98342320 0.95043265 0.94112984 0.98623899
Kubita sa cistocom ispod 1 - 1e-8: 6
Registar: 6 cistoce: 0.98972044 0.97953299 0.86788207 0.55319193 0.54764528 0.83651720
Kubita sa cistocom ispod 1 - 1e-8: 6
Registar: 7 cistoce: 0.97996451 0.97672907 0.99454437 0.70124058 0.65009922 0.87137005
Kubita sa cistocom ispod 1 - 1e-8: 6
Zavrsne Qiskit provere svih sedam registara su prosle.
Log legalne mase: -5.43813469
Log modelske verovatnoce NEXT: -11.60650826
Najvece marginalne verovatnoce: 0.37820 0.14540 0.13265 0.14418 0.14036 0.19781 0.30732
Promena raspodele izmedju stvarnih istorijskih konteksta: 0.02011730381
Raspon NEXT kombinacije: 38
Udeo istorijskih raspona manjih ili jednakih NEXT rasponu: 100.000%

CSV: loto7_4716_k89_loto_plus_1736.csv
Redova: 1735
SHA256: 56225ed0f0be7e2d3e30779804d8fa45e96ecfb9d14441eb2a731dd7fff38ae6
Trening prelaza: 1213
Validacionih prelaza: 260
Test prelaza: 261

Kandidat: prozor 1 regularizacija 0.001
UCENJE ZA VALIDACIJU
Trening prelaza: 1213
Regularizacija: 0.001
Obuka registra: 1
Status optimizatora: Maximum number of iterations has been exceeded.
Sacuvani gubitak: 0.90180833
Norma gradijenta sacuvane tacke: 1.2923974e-05
UPOZORENJE: konvergencija nije potvrdjena.
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95241407
Norma gradijenta sacuvane tacke: 8.6839704e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96290496
Norma gradijenta sacuvane tacke: 8.8417378e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96437870
Norma gradijenta sacuvane tacke: 8.3797301e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96205305
Norma gradijenta sacuvane tacke: 9.8162229e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95010670
Norma gradijenta sacuvane tacke: 9.3606022e-06
Obuka registra: 7
Status optimizatora: Maximum number of iterations has been exceeded.
Sacuvani gubitak: 0.90958797
Norma gradijenta sacuvane tacke: 5.1073873e-05
UPOZORENJE: konvergencija nije potvrdjena.
VALIDACIONI REZULTAT
Broj kola: 260
Srednji NLL: 19.08981802
Srednji broj pogodaka: 1.173077
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [66, 108, 63, 21, 2, 0, 0, 0]

Kandidat: prozor 1 regularizacija 0.01
UCENJE ZA VALIDACIJU
Trening prelaza: 1213
Regularizacija: 0.01
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95107157
Norma gradijenta sacuvane tacke: 9.3294208e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99632779
Norma gradijenta sacuvane tacke: 7.392531e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00517111
Norma gradijenta sacuvane tacke: 6.7756325e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00667343
Norma gradijenta sacuvane tacke: 9.0336518e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00532040
Norma gradijenta sacuvane tacke: 2.7978226e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98564111
Norma gradijenta sacuvane tacke: 7.9391402e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96679760
Norma gradijenta sacuvane tacke: 8.19081e-06
VALIDACIONI REZULTAT
Broj kola: 260
Srednji NLL: 18.76689369
Srednji broj pogodaka: 1.203846
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [63, 108, 64, 23, 2, 0, 0, 0]

Kandidat: prozor 3 regularizacija 0.001
UCENJE ZA VALIDACIJU
Trening prelaza: 1213
Regularizacija: 0.001
Obuka registra: 1
Status optimizatora: Maximum number of iterations has been exceeded.
Sacuvani gubitak: 0.89901091
Norma gradijenta sacuvane tacke: 9.7643703e-05
UPOZORENJE: konvergencija nije potvrdjena.
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95109346
Norma gradijenta sacuvane tacke: 9.4763718e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96125153
Norma gradijenta sacuvane tacke: 8.2148221e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96299521
Norma gradijenta sacuvane tacke: 9.4177515e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96040931
Norma gradijenta sacuvane tacke: 7.7891265e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.94917513
Norma gradijenta sacuvane tacke: 8.5667485e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.90690592
Norma gradijenta sacuvane tacke: 8.3197098e-06
VALIDACIONI REZULTAT
Broj kola: 260
Srednji NLL: 19.13959686
Srednji broj pogodaka: 1.288462
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [61, 99, 69, 27, 3, 1, 0, 0]

Kandidat: prozor 3 regularizacija 0.01
UCENJE ZA VALIDACIJU
Trening prelaza: 1213
Regularizacija: 0.01
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.94835395
Norma gradijenta sacuvane tacke: 7.417928e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99430908
Norma gradijenta sacuvane tacke: 7.3453958e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00308609
Norma gradijenta sacuvane tacke: 8.4858958e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00534873
Norma gradijenta sacuvane tacke: 7.098245e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00343724
Norma gradijenta sacuvane tacke: 7.6346456e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98433866
Norma gradijenta sacuvane tacke: 9.0777243e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96371087
Norma gradijenta sacuvane tacke: 7.7787744e-06
VALIDACIONI REZULTAT
Broj kola: 260
Srednji NLL: 18.61399369
Srednji broj pogodaka: 1.276923
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [49, 113, 77, 19, 2, 0, 0, 0]

Izabrani prozor: 3
Izabrana regularizacija: 0.01
UCENJE ZA ODVOJENI TEST
Trening prelaza: 1473
Regularizacija: 0.01
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95001389
Norma gradijenta sacuvane tacke: 7.8485377e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99434504
Norma gradijenta sacuvane tacke: 9.7571905e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00282312
Norma gradijenta sacuvane tacke: 2.3581286e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00590574
Norma gradijenta sacuvane tacke: 8.5157961e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00315669
Norma gradijenta sacuvane tacke: 4.9991944e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98429372
Norma gradijenta sacuvane tacke: 9.7188783e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96442901
Norma gradijenta sacuvane tacke: 7.7688913e-06
ODVOJENI TEST
Broj kola: 261
Srednji NLL: 18.23135056
Srednji broj pogodaka: 1.164751
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [62, 115, 65, 17, 2, 0, 0, 0]
ZAVRSNO UCENJE NA SVIM ISTORIJSKIM PRELAZIMA
Trening prelaza: 1734
Regularizacija: 0.01
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.94897222
Norma gradijenta sacuvane tacke: 9.763106e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99379307
Norma gradijenta sacuvane tacke: 5.1424246e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00313077
Norma gradijenta sacuvane tacke: 8.3165775e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00555500
Norma gradijenta sacuvane tacke: 3.4635363e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00286848
Norma gradijenta sacuvane tacke: 9.0783488e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98296680
Norma gradijenta sacuvane tacke: 6.055846e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96443710
Norma gradijenta sacuvane tacke: 7.8388123e-06
Cistoce pojedinacnih kubita u NEXT stanjima:
Registar: 1 cistoce: 0.99546198 0.97189917 0.90013936 0.76030054 0.69065677 0.80771548
Kubita sa cistocom ispod 1 - 1e-8: 6
Registar: 2 cistoce: 0.98768489 0.98278182 0.97063809 0.96181222 0.84507720 0.85524406
Kubita sa cistocom ispod 1 - 1e-8: 6
Registar: 3 cistoce: 0.99393915 0.98961281 0.98847786 0.91928484 0.87415049 0.93974696
Kubita sa cistocom ispod 1 - 1e-8: 6
Registar: 4 cistoce: 0.99210578 0.96438505 0.96571593 0.96178245 0.96074619 0.99241009
Kubita sa cistocom ispod 1 - 1e-8: 6
Registar: 5 cistoce: 0.98588572 0.98004051 0.98844995 0.94204365 0.91918539 0.96883298
Kubita sa cistocom ispod 1 - 1e-8: 6
Registar: 6 cistoce: 0.99850976 0.98451781 0.87709275 0.55500219 0.54324900 0.80507444
Kubita sa cistocom ispod 1 - 1e-8: 6
Registar: 7 cistoce: 0.96938826 0.96803405 0.99749971 0.69617163 0.63121005 0.83371802
Kubita sa cistocom ispod 1 - 1e-8: 6
Zavrsne Qiskit provere svih sedam registara su prosle.
Log legalne mase: -5.31102876
Log modelske verovatnoce NEXT: -12.10598487
Najvece marginalne verovatnoce: 0.36024 0.14732 0.12523 0.13344 0.13020 0.17542 0.30815
Promena raspodele izmedju stvarnih istorijskih konteksta: 0.08272040887
Raspon NEXT kombinacije: 38
Udeo istorijskih raspona manjih ili jednakih NEXT rasponu: 100.000%

PROVERA ISTIH NEXT KOMBINACIJA
loto7_4716_k89.csv
loto7_4716_k89_loto_2980.csv
Najveca razlika sirovih raspodela: 0.010055332135700562
Najveca razlika legalnih marginalnih raspodela: 0.02328518279415781




ZAVRSNE NEXT KOMBINACIJE

loto7_4716_k89.csv
NEXT: 1 x 16 y 25 z 39

loto7_4716_k89_loto_2980.csv
NEXT: 1 x 16 y 25 z 39

loto7_4716_k89_loto_plus_1736.csv
NEXT: 1 x 15 y 25 z 39
"""



"""
Q6-Q10	                            Osnovna ideja	                                                        Primena
Single Qubit Gates	                Rotacije, relativne faze i promene merne baze.	                        Učenje parametara jednokubitnih kapija. 
Multiple Qubits and Entanglement	Tenzorski proizvod, kontrolisane kapije i zapletenost.	                Učenje zapletenih stanja unutar šestokubitnih registara. 
Shor	                            Modularna eksponencijacija, procena faze i nalaženje reda.	            Posebna matematička veza sa istorijom kola. 
Bernstein-Vazirani	                Oporavak skrivenog niza iz linearnog binarnog orakla.	                Orakl izveden iz podataka koji zadovoljava taj uslov. 
Simon	                            Otkrivanje skrivene XOR simetrije funkcije.	                            Funkcija sa odgovarajućom strukturom preslikavanja. 


 6. Quantum_Single_Qubit_Gates.py
 7. Quantum_Entanglement.py
 8. Shor
 9. Bernstein-Vazirani
10. Simon

Za poslednja tri prvo treba definisati orakl iz istorije i proveriti uslove izvornog algoritma. 



--------
6. 

Quantum_Single_Qubit_Gates

Jednokubitna rotacija i fazna interferencija. 
Svaki šestokubitni registar ima proizvoljno stanje. 
Parametri kapija iz istorijskih prelaza kroz regresiju raspodele.



-------------
7. 

Quantum_Entanglement

Kontrolisane kapije i zapletenost. 
Između slojeva naučenih rotacija koriste se kontrolisane operacije. 
Provera redukovanih stanja pokazuje da li kolo zaista stvara zapletenost.

Rotacije zavisne od istorije pripremaju stanje, zatim CNOT lanac povezuje šest kubita, a završne rotacije menjaju mernu bazu. 
Provera zapletenosti preko čistoće redukovanih stanja i poređenje amplituda sa Qiskitom pre obuke.

Koristi naučene rotacije i CNOT lanac. 
Zapletenost se proverava unutar svakog šestokubitnog registra preko redukovanih matrica gustine. 
Redosled bitova prati Qiskit 1.4.

- smer i inverziju CNOT permutacija
- kompleksni gradijent kroz RY, RZ i CNOT
- faktor gradijenta Brierovog gubitka
- Bell i GHZ provere
- čistoća redukovanih stanja i Qiskit redosled bitova
- logaritamska normalizacija, globalni maksimum i hronološka podela 


Svaki kubit koristi niz RY → RZ → RY, koji pretvara relativnu fazu u promenu izlaznih verovatnoća. 
Prozor istorije bira se validacijom. 
Konačna kombinacija dobija se kao globalni maksimum raspodele nad svim legalnim izborima sedam brojeva.

Koristim sedam šestokubitnih registara i niz RY → RZ → RY na svakom kubitu. 
Prozore od jednog i tri prethodna kola biram validacijom. 
Optimizator uči parametre kapija iz cele trening istorije.


- izvodi verovatnoća po sva tri ugla RY → RZ → RY
- prefiksni i sufiksni proizvodi u gradijentu
- dimenzije parametara za oba prozora istorije
- redosled bitova prema Qiskitu
- logaritamska normalizacija i globalni izbor maksimuma
- granice treninga, validacije, testa i završnog učenja.



-------------
8. 

Shorov deo modularna eksponencijacija, QPE, inverzna QFT, procena faze i verizni ralomci za nalaženje reda. 

Orakl je \(f_a(x)=a^x\bmod 41\), uz izbor baze a hronološkom validacijom. 
Modul 41 omogućava modularni inverz za svaki broj od 1 do 39. 
QPE odredjuje red, a dodatni model uči amplitude koraka po modularnoj orbiti. 
Superpozicija periodičnih sopstvenih stanja modularnog operatora, čiju dimenziju određuje red dobijen QPE postupkom. 
Provera Shorove faktorizacije broja 15. 
Nalaženje reda koristiće 12 pomoćnih kubita i šest radnih kubita, a poređenje sa Qiskitom manju QPE konstrukciju.

Prediktivna dopuna uči superpoziciju periodičnih sopstvenih stanja modularnog operatora. 
Koristi modul 41, poslednje kolo kao početak orbite i celu istoriju prelaza za obuku. 
Bazu i broj koeficijenata bira validacijom.

QPE simulira 12 pomoćnih i šest radnih kubita. 
Manja QPE konstrukcija poredi se sa Qiskitom. 
Prediktivni deo koristi sedam šestokubitnih registara.



-------------
9.

Quantum_Bernstein_Vazirani 

Orakl, fazni povratni uticaj i Hadamard transformacije. 
Hadamard transformacije, linearni BV orakl i fazni povratni uticaj. 
Za Loto učim dodatnu fazu orakla iz istorijskih prelaza. 
Prethodno kolo je poznati ulaz. Buduća kombinacija se ne ugrađuje u orakl.

Koristim sedam registara sa po šest kubita i po jedan pomoćni kubit. 
Registri se simuliraju zasebno. 



----------------
10. 

Quantum_Simon

Sačuvaću orakl sa XOR periodom i rešavanje jednačina nad GF(2). 

Amplitude na parovima stanja \(x\) i \(x \oplus b\). 
Kandidati za XOR period biraju se iz treninga, a konačni model hronološkom validacijom. 
Simonovo kolo proverava period konstruisanog orakla. 

Dvostruki registar, orakl sa XOR periodom, Hadamard transformacije i rešavanje jednačina nad GF(2). 
Amplitude na Simonovim parovima stanja. 
Kandidati za period procenjuju se na treningu, a konačni model bira hronološkom validacijom.

Simonova provera koristi 12 kubita. 
Predikcija koristi sedam zasebnih registara sa po šest kubita. 
"""
