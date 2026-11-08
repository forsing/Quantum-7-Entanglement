"""
Quantum_Entanglement_V2.py

Python 3.11
Qiskit 1.4.4

KVANTNO KOLO JE ZADRZANO IZ V1
    RY(history) -> RZ(history)
    CNOT 0->1->2->3->4->5
    RY(readout)

Sedam zasebnih registara sa po sest kubita.
Zapletenost postoji unutar pojedinacnih registara.
Ne simulira se gust vektor od 42 kubita.

PROMENE PREMA V1
1. Zajednicki uslovni NLL umesto sedam nezavisnih Brier gubitaka.
2. Obuka koristi isti legalni prostor kombinacija kao NEXT.
3. Osnovni uglovi i zavrsne rotacije nemaju L2 kaznu.
4. L2 kazna obuhvata samo koeficijente istorijskih ulaza.
5. Centriranje i skaliranje uce se samo na odgovarajucem treningu.
6. Konstantni istorijski ulazi se uklanjaju.
7. Dva deterministicka pocetka optimizacije.
8. Dijagnostika uticaja istorije i razmaka prva dva maksimuma.

Raspodela legalne kombinacije:
    P(c | history) =
        product_j |psi_j(c_j | history)|^2 / Z(history)

gde je:
    0 <= c_0 < c_1 < ... < c_6 < 39.

Z i marginale racunaju se tacnim dinamickim programiranjem.

Gradijent prema logaritmu registarske verovatnoce:
    marginal_j(v) - indicator(v == target_j).

Kvantne amplitude daju modelsku raspodelu.
Klasicni optimizator uci parametre kvantnih kapija.
Klasicno dinamicko programiranje uslovljava raspodelu
i odredjuje globalni MAP.

Nema promene kombinacija radi medjusobnog razlikovanja CSV-ova.
Nema slucajnih kombinacija, dopunjavanja ili kozmetickih filtera.
"""

import csv
import hashlib
import itertools
import random
from pathlib import Path

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from scipy.optimize import minimize
from scipy.special import logsumexp


random.seed(39)
np.random.seed(39)


FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89_loto_2980.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89_loto_plus_1736.csv"),
)

N = 39
K = 7
QUBITS = 6
DIM = 1 << QUBITS

WINDOWS = (1, 3)
REGULARIZATIONS = (0.001, 0.01)

TRAIN_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85

BATCH_SIZE = 64
MAX_ITERATIONS = 200
GRADIENT_TOLERANCE = 1e-5
RESTARTS = 2

BASIS = np.arange(DIM, dtype=np.int64)
BITS = (
    BASIS[:, None]
    >> np.arange(QUBITS, dtype=np.int64)[None, :]
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


class DistributionError(ArithmeticError):
    pass


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def load_csv(path):
    raw = path.read_bytes()
    rows = []

    for line_number, row in enumerate(
        csv.reader(raw.decode("utf-8-sig").splitlines()),
        start=1,
    ):
        if not row or all(not cell.strip() for cell in row):
            continue

        require(
            len(row) == K,
            f"{path.name}, red {line_number}: potrebno je sedam kolona.",
        )

        try:
            values = [int(cell.strip()) for cell in row]
        except ValueError as exc:
            raise RuntimeError(
                f"{path.name}, red {line_number}: necelobrojna vrednost."
            ) from exc

        require(
            len(set(values)) == K
            and all(1 <= value <= N for value in values)
            and values == sorted(values),
            f"{path.name}, red {line_number}: nelegalna kombinacija.",
        )
        rows.append(values)

    require(len(rows) >= 5, f"{path.name}: nedovoljno kola.")

    return {
        "path": path,
        "data": np.asarray(rows, dtype=np.int64) - 1,
        "digest": hashlib.sha256(raw).hexdigest(),
    }


def raw_history_features(data, times, window):
    times = np.asarray(times, dtype=np.int64)

    require(
        window in WINDOWS
        and times.ndim == 1
        and len(times) > 0
        and np.all((times >= 1) & (times <= len(data))),
        "Neispravan istorijski ulaz.",
    )

    # Sedam stvarnih brojeva po prethodnom kolu.
    # Za starije lagove dodaje se oznaka dostupnosti.
    # Prvi lag je uvek dostupan i nema konstantnu oznaku.
    result = np.zeros(
        (len(times), K * window + window - 1),
        dtype=np.float64,
    )

    for lag in range(1, window + 1):
        source = times - lag
        available = source >= 0
        start = K * (lag - 1)

        result[available, start:start + K] = (
            data[source[available]].astype(np.float64) + 1.0 - 20.0
        ) / 19.0

        if lag >= 2:
            result[:, K * window + lag - 2] = available.astype(float)

    return result


def fit_feature_transform(raw, training_rows):
    require(
        1 <= training_rows <= len(raw),
        "Neispravna granica za ucenje skaliranja.",
    )

    training = raw[:training_rows]
    mean = training.mean(axis=0)
    scale = training.std(axis=0)
    keep = scale > 1e-12

    return {
        "mean": mean,
        "scale": scale,
        "keep": keep,
    }


def transform_features(raw, transform):
    keep = transform["keep"]

    result = np.ones(
        (len(raw), 1 + int(np.sum(keep))),
        dtype=np.float64,
    )

    if np.any(keep):
        result[:, 1:] = (
            raw[:, keep] - transform["mean"][keep]
        ) / transform["scale"][keep]

    require(np.isfinite(result).all(), "Nekonacne karakteristike.")
    return result


def parameter_count(feature_count):
    return QUBITS * (2 * feature_count + 1)


def split_parameters(theta, feature_count):
    matrix = np.asarray(theta, dtype=float).reshape(
        QUBITS, 2 * feature_count + 1
    )
    return (
        matrix[:, :feature_count],
        matrix[:, feature_count:2 * feature_count],
        matrix[:, -1],
    )


def regularization_mask(feature_count):
    mask = np.zeros(
        (K, QUBITS, 2 * feature_count + 1),
        dtype=np.float64,
    )

    # Intercept RY je kolona 0.
    # Intercept RZ je kolona feature_count.
    # Readout je poslednja kolona.
    mask[:, :, 1:feature_count] = 1.0
    mask[:, :, feature_count + 1:2 * feature_count] = 1.0
    return mask.ravel()


def initial_parameters(feature_count, restart):
    result = np.zeros(
        (K, QUBITS, 2 * feature_count + 1),
        dtype=np.float64,
    )

    for slot in range(K):
        index = np.arange(
            QUBITS * feature_count, dtype=np.float64
        ).reshape(QUBITS, feature_count)

        phase = index + 1.0 + 0.31 * slot + 0.73 * restart

        result[slot, :, :feature_count] = (
            0.06 * np.sin(phase) / np.sqrt(feature_count)
        )
        result[slot, :, feature_count:2 * feature_count] = (
            0.06 * np.cos(phase) / np.sqrt(feature_count)
        )

        qubits = np.arange(1, QUBITS + 1, dtype=np.float64)

        result[slot, :, 0] = (
            np.pi / 2.0
            - 0.35 * restart
            + 0.10 * np.sin(qubits + 0.2 * slot)
        )
        result[slot, :, feature_count] = (
            0.50
            + 0.25 * restart
            + 0.10 * np.cos(qubits + 0.3 * slot)
        )
        result[slot, :, -1] = (
            0.35
            + 0.30 * restart
            + 0.10 * np.sin(qubits + 0.4 * slot)
        )

    return result.ravel()


def apply_ry(values, angles, qubit):
    zero = ZERO_INDICES[qubit]
    one = ONE_INDICES[qubit]
    cosine = np.cos(angles / 2.0)[:, None]
    sine = np.sin(angles / 2.0)[:, None]

    result = values.copy()
    result[:, zero] = (
        cosine * values[:, zero] - sine * values[:, one]
    )
    result[:, one] = (
        sine * values[:, zero] + cosine * values[:, one]
    )
    return result


def apply_rz(values, angles, qubit):
    return values * np.exp(
        -0.5j * angles[:, None] * Z_SIGNS[qubit][None, :]
    )


def forward_register(theta, features, record=False):
    input_weights, phase_weights, readout = split_parameters(
        theta, features.shape[1]
    )
    input_angles = features @ input_weights.T
    phase_angles = features @ phase_weights.T

    values = np.zeros((len(features), DIM), dtype=np.complex128)
    values[:, 0] = 1.0
    cache = []

    for qubit in range(QUBITS):
        angles = input_angles[:, qubit]

        if record:
            cache.append(("ry", qubit, "input", angles, values))

        values = apply_ry(values, angles, qubit)
        angles = phase_angles[:, qubit]

        if record:
            cache.append(("rz", qubit, "phase", angles, values))

        values = apply_rz(values, angles, qubit)

    for control in range(QUBITS - 1):
        if record:
            cache.append(("cx", control, None, None, None))

        values = values[:, CNOT_PERMUTATIONS[control]]

    for qubit in range(QUBITS):
        angles = np.full(len(features), readout[qubit])

        if record:
            cache.append(("ry", qubit, "readout", angles, values))

        values = apply_ry(values, angles, qubit)

    require(
        np.isfinite(values).all()
        and np.allclose(
            np.sum(np.abs(values) ** 2, axis=1),
            1.0,
            atol=2e-10,
            rtol=0.0,
        ),
        "Kvantni registar nije normalizovan.",
    )
    return values, cache


def backward_register(adjoint, cache, features):
    feature_count = features.shape[1]
    gradient = np.zeros(
        (QUBITS, 2 * feature_count + 1),
        dtype=np.float64,
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
                sine * before[:, zero] + cosine * before[:, one]
            )
            derivative_one = 0.5 * (
                cosine * before[:, zero] - sine * before[:, one]
            )

            angle_gradient = np.real(np.sum(
                values[:, zero].conj() * derivative_zero
                + values[:, one].conj() * derivative_one,
                axis=1,
            ))
            values = apply_ry(values, -angles, qubit)

        elif gate == "rz":
            after = apply_rz(before, angles, qubit)
            derivative = (
                -0.5j * Z_SIGNS[qubit][None, :] * after
            )
            angle_gradient = np.real(np.sum(
                values.conj() * derivative, axis=1
            ))
            values = apply_rz(values, -angles, qubit)

        else:
            raise RuntimeError("Nepoznata kapija.")

        if group == "input":
            gradient[qubit, :feature_count] += (
                angle_gradient @ features
            )
        elif group == "phase":
            gradient[qubit, feature_count:2 * feature_count] += (
                angle_gradient @ features
            )
        elif group == "readout":
            gradient[qubit, -1] += angle_gradient.sum()
        else:
            raise RuntimeError("Nepoznata grupa parametara.")

    return gradient.ravel()


def log_amplitudes(amplitudes):
    absolute = np.abs(amplitudes)
    result = np.full(absolute.shape, -np.inf, dtype=np.float64)
    positive = absolute > 0.0
    result[positive] = 2.0 * np.log(absolute[positive])
    return result


def conditional_statistics(log_weights, targets=None):
    """
    Ulaz: batch x broj_registara x broj_stanja.

    Vraca tacan normalizator i marginale rastucih kombinacija.
    """
    logs = np.asarray(log_weights, dtype=np.float64).copy()

    require(
        logs.ndim == 3
        and not np.isnan(logs).any()
        and not np.isposinf(logs).any(),
        "Neispravni logaritmi zajednicke raspodele.",
    )

    batch, slots, states = logs.shape
    require(
        batch > 0 and 1 <= slots <= states,
        "Neispravne dimenzije zajednicke raspodele.",
    )

    for slot in range(slots):
        logs[:, slot, :slot] = -np.inf
        logs[:, slot, states - slots + slot + 1:] = -np.inf

    forward = np.full_like(logs, -np.inf)
    backward = np.full_like(logs, -np.inf)
    forward[:, 0] = logs[:, 0]

    for slot in range(1, slots):
        prefix = np.logaddexp.accumulate(
            forward[:, slot - 1], axis=1
        )
        strict_prefix = np.concatenate(
            [np.full((batch, 1), -np.inf), prefix[:, :-1]],
            axis=1,
        )
        forward[:, slot] = logs[:, slot] + strict_prefix

    log_z = logsumexp(forward[:, -1], axis=1)

    if not np.isfinite(log_z).all():
        raise DistributionError("Nema pozitivne legalne mase.")

    backward[:, -1] = 0.0

    for slot in range(slots - 2, -1, -1):
        next_values = logs[:, slot + 1] + backward[:, slot + 1]
        suffix = np.logaddexp.accumulate(
            next_values[:, ::-1], axis=1
        )[:, ::-1]

        backward[:, slot] = np.concatenate(
            [suffix[:, 1:], np.full((batch, 1), -np.inf)],
            axis=1,
        )

    marginals = np.exp(
        forward + backward - log_z[:, None, None]
    )

    require(
        np.all(log_z <= 2e-8)
        and np.allclose(
            marginals.sum(axis=2),
            1.0,
            atol=2e-8,
            rtol=0.0,
        ),
        "Zajednicka raspodela nije normalizovana.",
    )

    losses = None

    if targets is not None:
        targets = np.asarray(targets, dtype=np.int64)

        require(
            targets.shape == (batch, slots)
            and np.all((targets >= 0) & (targets < states))
            and np.all(np.diff(targets, axis=1) > 0),
            "Neispravni ciljevi zajednicke raspodele.",
        )

        target_log_mass = logs[
            np.arange(batch)[:, None],
            np.arange(slots)[None, :],
            targets,
        ].sum(axis=1)

        losses = log_z - target_log_mass

        require(
            not np.isnan(losses).any()
            and np.all(losses >= -2e-8),
            "Neispravan zajednicki NLL.",
        )
        losses = np.maximum(losses, 0.0)

    return {
        "logs": logs,
        "log_z": log_z,
        "marginals": marginals,
        "losses": losses,
    }


def joint_distribution(log_weights):
    statistics = conditional_statistics(
        np.asarray(log_weights)[None, :, :]
    )
    logs = statistics["logs"][0]
    log_z = float(statistics["log_z"][0])
    slots, states = logs.shape

    maximum = np.full((slots, states), -np.inf)
    parents = np.full((slots, states), -1, dtype=np.int64)
    maximum[0] = logs[0]

    # Dva najbolja razlicita puta po zavrsnom stanju.
    top_two = np.full((states, 2), -np.inf)
    top_two[:, 0] = logs[0]

    for slot in range(1, slots):
        next_two = np.full_like(top_two, -np.inf)
        running_index = 0
        running_scores = np.array([-np.inf, -np.inf])

        for value in range(1, states):
            previous = value - 1

            if (
                maximum[slot - 1, previous]
                > maximum[slot - 1, running_index]
            ):
                running_index = previous

            maximum[slot, value] = (
                logs[slot, value]
                + maximum[slot - 1, running_index]
            )
            parents[slot, value] = running_index

            candidates = np.concatenate([
                running_scores, top_two[previous]
            ])
            running_scores = np.sort(candidates)[-2:][::-1]
            next_two[value] = logs[slot, value] + running_scores

        top_two = next_two

    path = np.empty(slots, dtype=np.int64)
    path[-1] = int(np.argmax(maximum[-1]))

    for slot in range(slots - 1, 0, -1):
        path[slot - 1] = parents[slot, path[slot]]

    scores = np.sort(top_two.ravel())[-2:][::-1]
    log_mode = float(logs[np.arange(slots), path].sum())

    require(
        np.all(np.diff(path) > 0)
        and np.isfinite(log_mode)
        and log_mode <= log_z + 2e-8
        and np.isclose(log_mode, scores[0], atol=2e-8),
        "Neispravan globalni maksimum.",
    )

    return {
        "path": path,
        "logs": logs,
        "log_z": log_z,
        "marginals": statistics["marginals"][0],
        "log_mode_probability": log_mode - log_z,
        "mode_gap": float(scores[0] - scores[1]),
    }


def objective(theta, features, targets, weight):
    feature_count = features.shape[1]
    count = parameter_count(feature_count)
    parameters = np.asarray(theta).reshape(K, count)

    gradient = np.zeros_like(parameters)
    loss_sum = 0.0

    for start in range(0, len(targets), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(targets))
        batch_features = features[start:stop]
        batch_targets = targets[start:stop]

        states = []
        caches = []

        for slot in range(K):
            state, cache = forward_register(
                parameters[slot], batch_features, record=True
            )
            states.append(state)
            caches.append(cache)

        log_weights = np.stack(
            [log_amplitudes(state[:, :N]) for state in states],
            axis=1,
        )

        try:
            statistics = conditional_statistics(
                log_weights, batch_targets
            )
        except DistributionError:
            return np.inf, np.zeros_like(theta)

        if not np.isfinite(statistics["losses"]).all():
            return np.inf, np.zeros_like(theta)

        loss_sum += float(statistics["losses"].sum())
        log_probability_gradient = statistics["marginals"].copy()

        log_probability_gradient[
            np.arange(stop - start)[:, None],
            np.arange(K)[None, :],
            batch_targets,
        ] -= 1.0

        for slot in range(K):
            state = states[slot]
            adjoint = np.zeros_like(state)
            denominator = state[:, :N].conj()
            numerator = 2.0 * log_probability_gradient[:, slot]

            # Nema dodavanja epsilona ili menjanja raspodele.
            # Na nultoj amplitudi bez ciljne mase doprinos je nula.
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                np.divide(
                    numerator,
                    denominator,
                    out=adjoint[:, :N],
                    where=np.abs(denominator) > 0.0,
                )

            if not np.isfinite(adjoint).all():
                return np.inf, np.zeros_like(theta)

            gradient[slot] += backward_register(
                adjoint, caches[slot], batch_features
            )

    mask = regularization_mask(feature_count)
    denominator = max(1.0, float(mask.sum()))

    penalty = weight * np.dot(mask * theta, theta) / denominator
    penalty_gradient = (
        2.0 * weight * mask * theta / denominator
    )

    loss = loss_sum / len(targets) + penalty
    full_gradient = gradient.ravel() / len(targets) + penalty_gradient

    require(
        np.isfinite(loss) and np.isfinite(full_gradient).all(),
        "Nekonacan gubitak ili gradijent.",
    )
    return float(loss), full_gradient


def fit_models(features, targets, weight, label):
    require(
        len(features) == len(targets) and len(targets) > 0,
        "Neispravan trening skup.",
    )

    print(label, flush=True)
    print(f"Trening prelaza: {len(targets)}")
    print(f"Broj karakteristika sa interceptom: {features.shape[1]}")
    print(f"Regularizacija istorijskih koeficijenata: {weight}")

    best = None

    for restart in range(RESTARTS):
        tracked = {"loss": np.inf, "theta": None}

        def tracked_objective(candidate):
            value, gradient = objective(
                candidate, features, targets, weight
            )

            if np.isfinite(value) and value < tracked["loss"]:
                tracked["loss"] = value
                tracked["theta"] = candidate.copy()

            return value, gradient

        print(
            f"Deterministicki pocetak {restart + 1}/{RESTARTS}",
            flush=True,
        )

        result = minimize(
            tracked_objective,
            initial_parameters(features.shape[1], restart),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": MAX_ITERATIONS,
                "gtol": GRADIENT_TOLERANCE,
                "ftol": 1e-10,
                "maxls": 30,
                "maxcor": 10,
            },
        )

        require(
            tracked["theta"] is not None,
            "Optimizator nije nasao konacan kandidat.",
        )

        value, gradient = objective(
            tracked["theta"], features, targets, weight
        )
        gradient_norm = float(np.max(np.abs(gradient)))

        print(f"Sacuvani zajednicki NLL+L2: {value:.9f}")
        print(f"Max|grad|: {gradient_norm:.3e}")
        print(f"Status: {result.message}")

        if gradient_norm > GRADIENT_TOLERANCE:
            print("Napomena: mali gradijent nije potvrdjen.")

        if best is None or value < best["loss"]:
            best = {
                "loss": value,
                "theta": tracked["theta"].copy(),
            }

    count = parameter_count(features.shape[1])
    return best["theta"].reshape(K, count).copy()


def predict_logs(models, features):
    logs = np.empty((len(features), K, DIM), dtype=np.float64)

    for start in range(0, len(features), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(features))

        for slot in range(K):
            states, _ = forward_register(
                models[slot], features[start:stop]
            )
            logs[start:stop, slot] = log_amplitudes(states)

    require(
        np.allclose(
            logsumexp(logs, axis=2),
            0.0,
            atol=2e-10,
            rtol=0.0,
        ),
        "Registarske raspodele nisu normalizovane.",
    )
    return logs


def evaluate(models, features, targets):
    losses = []
    hits = []

    for start in range(0, len(targets), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(targets))
        logs = predict_logs(models, features[start:stop])[:, :, :N]
        batch_targets = targets[start:stop]
        statistics = conditional_statistics(logs, batch_targets)
        losses.extend(statistics["losses"].tolist())

        for row, target in enumerate(batch_targets):
            joint = joint_distribution(logs[row])
            hits.append(
                len(set(joint["path"].tolist()) & set(target.tolist()))
            )

    require(len(losses) > 0, "Prazan evaluacioni skup.")
    losses = np.asarray(losses)

    return {
        "nll": float(losses.mean()),
        "hits": np.asarray(hits, dtype=np.int64),
        "zero_targets": int(np.isposinf(losses).sum()),
    }


def print_metrics(label, metrics):
    print(label)
    print(f"Srednji uslovni NLL: {metrics['nll']:.9f}")
    print(f"Srednji broj pogodaka: {metrics['hits'].mean():.6f}")
    print(f"Ciljevi sa nultom masom: {metrics['zero_targets']}")
    print(
        "Broj kola po pogocima 0..7: "
        + str(np.bincount(metrics["hits"], minlength=K + 1).tolist())
    )


def verify_qiskit(theta, features):
    expected, _ = forward_register(theta, features)
    input_weights, phase_weights, readout = split_parameters(
        theta, features.shape[1]
    )
    input_angles = features @ input_weights.T
    phase_angles = features @ phase_weights.T

    actual_states = []
    purities = []

    for row in range(len(features)):
        circuit = QuantumCircuit(QUBITS)

        for qubit in range(QUBITS):
            circuit.ry(float(input_angles[row, qubit]), qubit)
            circuit.rz(float(phase_angles[row, qubit]), qubit)

        for control in range(QUBITS - 1):
            circuit.cx(control, control + 1)

        for qubit in range(QUBITS):
            circuit.ry(float(readout[qubit]), qubit)

        state = Statevector.from_instruction(circuit)
        actual = np.asarray(state.data)

        require(
            np.allclose(
                actual, expected[row],
                atol=2e-10, rtol=2e-10,
            ),
            "Qiskit i analiticki simulator daju razlicite amplitude.",
        )

        row_purities = []

        for qubit in range(QUBITS):
            reduced = partial_trace(
                state,
                [other for other in range(QUBITS) if other != qubit],
            ).data
            purity = float(np.trace(reduced @ reduced).real)

            zero = actual[ZERO_INDICES[qubit]]
            one = actual[ONE_INDICES[qubit]]
            reference = (
                np.vdot(zero, zero).real ** 2
                + np.vdot(one, one).real ** 2
                + 2.0 * abs(np.vdot(one, zero)) ** 2
            )

            require(
                0.5 - 2e-9 <= purity <= 1.0 + 2e-9
                and np.isclose(purity, reference, atol=2e-10),
                "Neispravna redukovana cistoca.",
            )
            row_purities.append(purity)

        actual_states.append(actual)
        purities.append(row_purities)

    return np.asarray(actual_states), np.asarray(purities)


def self_checks():
    fixture = np.array([
        [0, 4, 9, 13, 20, 30, 38],
        [1, 6, 10, 14, 22, 31, 37],
        [0, 5, 12, 16, 24, 33, 38],
        [2, 7, 11, 18, 23, 29, 36],
        [1, 8, 14, 19, 26, 32, 37],
        [3, 9, 15, 21, 27, 34, 38],
    ], dtype=np.int64)
    times = np.arange(1, len(fixture))

    # Bell stanje proverava CNOT i little-endian redosled.
    bell = np.zeros((1, DIM), dtype=np.complex128)
    bell[0, 0] = 1.0
    bell = apply_ry(bell, np.array([np.pi / 2.0]), 0)
    bell = bell[:, CNOT_PERMUTATIONS[0]]

    circuit = QuantumCircuit(QUBITS)
    circuit.h(0)
    circuit.cx(0, 1)

    require(
        np.allclose(
            bell[0],
            Statevector.from_instruction(circuit).data,
            atol=1e-12,
            rtol=0.0,
        ),
        "Bell provera nije prosla.",
    )

    for window in WINDOWS:
        raw = raw_history_features(fixture, times, window)
        transform = fit_feature_transform(raw, 3)
        features = transform_features(raw, transform)
        theta = initial_parameters(features.shape[1], 0)

        count = parameter_count(features.shape[1])
        models = theta.reshape(K, count)

        for slot in range(K):
            verify_qiskit(models[slot], features[:2])

        value, analytic = objective(
            theta, features, fixture[1:], 0.003
        )
        require(np.isfinite(value), "Neispravan pocetni NLL.")

        # Koordinatne provere i provere usmerenih izvoda.
        selected = set(
            np.linspace(0, len(theta) - 1, 24, dtype=int).tolist()
        )

        for slot in range(K):
            selected.add(slot * count)
            selected.add(slot * count + features.shape[1])
            selected.add((slot + 1) * count - 1)

        step = 1e-6

        for index in sorted(selected):
            upper = theta.copy()
            lower = theta.copy()
            upper[index] += step
            lower[index] -= step

            numeric = (
                objective(upper, features, fixture[1:], 0.003)[0]
                - objective(lower, features, fixture[1:], 0.003)[0]
            ) / (2.0 * step)

            require(
                np.isclose(
                    analytic[index], numeric,
                    atol=3e-5, rtol=3e-4,
                ),
                "Koordinatna provera zajednickog gradijenta nije prosla.",
            )

        for direction_index in range(3):
            direction = np.sin(
                (np.arange(len(theta)) + 1.0)
                * (0.31 + 0.17 * direction_index)
            )
            direction /= np.linalg.norm(direction)

            numeric = (
                objective(
                    theta + step * direction,
                    features, fixture[1:], 0.003,
                )[0]
                - objective(
                    theta - step * direction,
                    features, fixture[1:], 0.003,
                )[0]
            ) / (2.0 * step)

            require(
                np.isclose(
                    analytic @ direction, numeric,
                    atol=3e-5, rtol=3e-4,
                ),
                "Usmerena provera zajednickog gradijenta nije prosla.",
            )

        # Menjanje kasnijih kola ne sme promeniti ranije ulaze
        # niti skaliranje nauceno na prva tri prelaza.
        changed = fixture.copy()
        changed[4:] = changed[4:, ::-1]

        changed_raw = raw_history_features(changed, times, window)
        changed_transform = fit_feature_transform(changed_raw, 3)

        require(
            np.array_equal(raw[:3], changed_raw[:3])
            and np.array_equal(
                transform["keep"], changed_transform["keep"]
            )
            and np.allclose(
                transform["mean"], changed_transform["mean"]
            )
            and np.allclose(
                transform["scale"], changed_transform["scale"]
            ),
            "Buduci podaci uticu na raniji trening.",
        )

    toy = np.exp(
        0.5 * np.sin(np.arange(15, dtype=float).reshape(3, 5))
    )
    toy /= toy.sum(axis=1, keepdims=True)
    logs = np.log(toy)
    joint = joint_distribution(logs)

    combinations = list(itertools.combinations(range(5), 3))
    scores = np.array([
        logs[np.arange(3), combination].sum()
        for combination in combinations
    ])
    brute_marginals = np.zeros_like(toy)

    for combination, mass in zip(
        combinations, np.exp(scores - logsumexp(scores))
    ):
        brute_marginals[np.arange(3), combination] += mass

    sorted_scores = np.sort(scores)[::-1]

    require(
        np.isclose(joint["log_z"], logsumexp(scores), atol=1e-12)
        and np.isclose(
            logs[np.arange(3), joint["path"]].sum(),
            sorted_scores[0],
            atol=1e-12,
        )
        and np.isclose(
            joint["mode_gap"],
            sorted_scores[0] - sorted_scores[1],
            atol=1e-12,
        )
        and np.allclose(
            joint["marginals"], brute_marginals,
            atol=1e-12, rtol=0.0,
        ),
        "Potpuna provera DP raspodele nije prosla.",
    )

    sole = np.full((3, 5), -np.inf)
    sole[np.arange(3), [0, 2, 4]] = 0.0
    sole_joint = joint_distribution(sole)

    require(
        np.array_equal(sole_joint["path"], [0, 2, 4])
        and np.isclose(sole_joint["log_z"], 0.0)
        and np.isposinf(sole_joint["mode_gap"]),
        "Provera jedine legalne kombinacije nije prosla.",
    )

    impossible = np.full((3, 5), -np.inf)
    impossible[:, 0] = 0.0

    try:
        joint_distribution(impossible)
    except DistributionError:
        pass
    else:
        raise RuntimeError("Nulta legalna masa nije prijavljena.")

    require(
        np.isfinite(log_amplitudes(np.array([1e-180 + 0j]))[0]),
        "Izgubljena mala amplituda.",
    )

    print("Ugradjene Qiskit, gradijent, hronologija i DP provere su prosle.")


def longest_consecutive(values):
    longest = 1
    current = 1

    for difference in np.diff(values):
        current = current + 1 if difference == 1 else 1
        longest = max(longest, current)

    return longest


def process_dataset(dataset):
    path = dataset["path"]
    data = dataset["data"]
    targets = data[1:]
    times = np.arange(1, len(data))

    train_end = int(TRAIN_FRACTION * len(targets))
    validation_end = int(VALIDATION_END_FRACTION * len(targets))

    require(
        0 < train_end < validation_end < len(targets),
        "Nedovoljno podataka za hronolosku podelu.",
    )

    print()
    print("=" * 72)
    print(path.name)
    print(f"Stvarni broj kola: {len(data)}")
    print(f"SHA256: {dataset['digest']}")
    print(f"Trening prelaza: {train_end}")
    print(f"Validacionih prelaza: {validation_end - train_end}")
    print(f"Test prelaza: {len(targets) - validation_end}")

    best = None

    for window in WINDOWS:
        raw = raw_history_features(data, times, window)
        transform = fit_feature_transform(raw, train_end)
        features = transform_features(raw, transform)

        for weight in REGULARIZATIONS:
            label = f"VALIDACIJA prozor={window}, L2={weight}"
            models = fit_models(
                features[:train_end],
                targets[:train_end],
                weight,
                label,
            )

            try:
                metrics = evaluate(
                    models,
                    features[train_end:validation_end],
                    targets[train_end:validation_end],
                )
            except DistributionError as exc:
                print(f"Kandidat odbijen: {exc}")
                continue

            print_metrics(label, metrics)

            if np.isfinite(metrics["nll"]) and (
                best is None or metrics["nll"] < best["nll"]
            ):
                best = {
                    "window": window,
                    "weight": weight,
                    "nll": metrics["nll"],
                }

    require(best is not None, "Nema validnog modela.")

    window = best["window"]
    weight = best["weight"]
    raw = raw_history_features(data, times, window)

    print()
    print(f"Izabrani prozor: {window}")
    print(f"Izabrana regularizacija: {weight}")

    # Skaliranje se ponovo uci samo na treningu za test.
    test_transform = fit_feature_transform(raw, validation_end)
    test_features = transform_features(raw, test_transform)

    test_models = fit_models(
        test_features[:validation_end],
        targets[:validation_end],
        weight,
        "UCENJE ZA ODVOJENI TEST",
    )
    test_metrics = evaluate(
        test_models,
        test_features[validation_end:],
        targets[validation_end:],
    )
    print_metrics("ODVOJENI HRONOLOSKI TEST", test_metrics)

    # Hiperparametri su zakljucani pre testa.
    # Zavrsna obuka i skaliranje sada koriste celu istoriju.
    final_transform = fit_feature_transform(raw, len(raw))
    final_features = transform_features(raw, final_transform)

    final_models = fit_models(
        final_features,
        targets,
        weight,
        "ZAVRSNO UCENJE NA SVIM PRELAZIMA",
    )

    next_raw = raw_history_features(
        data, np.array([len(data)]), window
    )
    next_features = transform_features(next_raw, final_transform)

    numerical_logs = predict_logs(final_models, next_features)[0]
    next_logs = np.empty((K, DIM), dtype=float)

    print("Cistoce pojedinacnih kubita:")

    for slot in range(K):
        states, purities = verify_qiskit(
            final_models[slot], next_features
        )
        next_logs[slot] = log_amplitudes(states[0])

        print(
            f"Registar {slot + 1}: "
            + " ".join(f"{value:.8f}" for value in purities[0])
        )

        if np.all(purities[0] >= 1.0 - 1e-8):
            print("Napomena: na ovom ulazu nije utvrdjena zapletenost.")

    require(
        np.allclose(
            np.exp(next_logs),
            np.exp(numerical_logs),
            atol=2e-10,
            rtol=2e-9,
        ),
        "Zavrsne Qiskit raspodele nisu dosledne.",
    )

    joint = joint_distribution(next_logs[:, :N])
    combination = joint["path"] + 1

    require(
        len(set(combination.tolist())) == K
        and np.all((combination >= 1) & (combination <= N)),
        "Nelegalna NEXT kombinacija.",
    )

    # Istorijski konteksti i ulaz sa srednjim trening karakteristikama.
    probe_times = np.unique(
        np.array([1, max(1, len(data) // 2), len(data)])
    )
    probe_features = transform_features(
        raw_history_features(data, probe_times, window),
        final_transform,
    )

    probes = predict_logs(final_models, probe_features)
    raw_variation = float(np.max(np.abs(
        np.exp(probes) - np.exp(next_logs)[None, :, :]
    )))

    centered = np.zeros_like(next_features)
    centered[:, 0] = 1.0
    centered_logs = predict_logs(final_models, centered)[0]
    centered_joint = joint_distribution(centered_logs[:, :N])

    marginal_effect = float(np.max(np.abs(
        joint["marginals"] - centered_joint["marginals"]
    )))

    print(f"Log legalne mase: {joint['log_z']:.9f}")
    print(
        f"Log uslovne mase NEXT: "
        f"{joint['log_mode_probability']:.9f}"
    )
    print(
        f"Log odnos prvog i drugog maksimuma: "
        f"{joint['mode_gap']:.9f}"
    )
    print(f"Promena kroz istorijske kontekste: {raw_variation:.9e}")
    print(
        f"Uticaj trenutne istorije na legalne marginale: "
        f"{marginal_effect:.9e}"
    )

    if marginal_effect < 1e-8:
        print("Upozorenje: veoma slab uticaj trenutne istorije.")

    if joint["log_z"] < np.log(1e-10):
        print("Upozorenje: veoma mala legalna masa pre uslovljavanja.")

    if np.array_equal(combination, data[-1] + 1):
        print("Upozorenje: NEXT ponavlja poslednje kolo.")

    run = longest_consecutive(combination)

    if run >= 5:
        print(f"Upozorenje: niz od {run} uzastopnih brojeva.")

    span = int(combination[-1] - combination[0])
    historical_spans = data[:, -1] - data[:, 0]
    percentile = 100.0 * float(np.mean(historical_spans <= span))

    print(f"Raspon NEXT kombinacije: {span}")
    print(f"Udeo istorijskih raspona <= NEXT raspon: {percentile:.3f}%")

    if percentile <= 1.0:
        print("Upozorenje: izrazito grupisana NEXT kombinacija.")

    return {
        "name": path.name,
        "digest": dataset["digest"],
        "next": combination,
        "probabilities": np.exp(next_logs),
        "marginals": joint["marginals"],
        "logs": joint["logs"],
        "log_z": joint["log_z"],
        "mode_gap": joint["mode_gap"],
    }


def main():
    print("Quantum Entanglement V2")
    print(f"Qiskit: {qiskit.__version__}")
    print("Sedam zasebnih registara sa po sest kubita.")
    print("Kolo iz V1: RY, RZ, CNOT lanac, zavrsni RY.")
    print("Obuka zajednicke raspodele legalnih kombinacija.")
    print("Regularizuju se samo istorijski koeficijenti.")
    print("Skaliranje se uci samo na odgovarajucem treningu.")
    print("Dva deterministicka pocetka optimizacije.")
    print("Kombinacije se ne menjaju radi medjusobnog razlikovanja.")
    print()

    self_checks()

    results = [
        process_dataset(load_csv(path))
        for path in FILES
    ]

    print()
    print("=" * 72)
    print("POREDJENJE NEZAVISNIH MODELA")

    for first, second in itertools.combinations(results, 2):
        print()
        print(first["name"])
        print(second["name"])

        if first["digest"] == second["digest"]:
            print("Upozorenje: ulazni fajlovi imaju isti SHA256.")

        print(
            "Najveca razlika sirovih raspodela: "
            f"{np.max(np.abs(first['probabilities'] - second['probabilities'])):.9e}"
        )
        print(
            "Najveca razlika legalnih marginala: "
            f"{np.max(np.abs(first['marginals'] - second['marginals'])):.9e}"
        )

        if np.array_equal(first["next"], second["next"]):
            print("Upozorenje: razliciti modeli imaju isti NEXT maksimum.")
            print(
                "Razmaci prvog i drugog maksimuma: "
                f"{first['mode_gap']:.9f}, {second['mode_gap']:.9f}"
            )
        else:
            overlap = len(
                set(first["next"].tolist())
                & set(second["next"].tolist())
            )
            print(f"Zajednickih NEXT brojeva: {overlap}")

    print()
    print("=" * 72)

    for result in results:
        numbers = " ".join(
            str(int(value)) for value in result["next"]
        )
        print(f"{result['name']} NEXT: {numbers}")


if __name__ == "__main__":
    main()



"""
Quantum Entanglement V2
Qiskit: 1.4.4
Sedam zasebnih registara sa po sest kubita.
Kolo iz V1: RY, RZ, CNOT lanac, zavrsni RY.
Obuka zajednicke raspodele legalnih kombinacija.
Regularizuju se samo istorijski koeficijenti.
Skaliranje se uci samo na odgovarajucem treningu.
Dva deterministicka pocetka optimizacije.
Kombinacije se ne menjaju radi medjusobnog razlikovanja.

Ugradjene Qiskit, gradijent, hronologija i DP provere su prosle.

========================================================================
loto7_4716_k89.csv
Stvarni broj kola: 4716
SHA256: deb02ea741f500e3270402730ed89565f1c9b4618d53e5a08fea2b7c2c2af31a
Trening prelaza: 3300
Validacionih prelaza: 707
Test prelaza: 708
VALIDACIJA prozor=1, L2=0.001
Trening prelaza: 3300
Broj karakteristika sa interceptom: 8
Regularizacija istorijskih koeficijenata: 0.001
Deterministicki pocetak 1/2
Sacuvani zajednicki NLL+L2: 16.420822605
Max|grad|: 1.012e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Sacuvani zajednicki NLL+L2: 16.408343548
Max|grad|: 9.540e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
VALIDACIJA prozor=1, L2=0.001
Srednji uslovni NLL: 16.689042122
Srednji broj pogodaka: 1.295615
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [145, 276, 223, 58, 5, 0, 0, 0]
VALIDACIJA prozor=1, L2=0.01
Trening prelaza: 3300
Broj karakteristika sa interceptom: 8
Regularizacija istorijskih koeficijenata: 0.01
Deterministicki pocetak 1/2
Sacuvani zajednicki NLL+L2: 16.422424260
Max|grad|: 5.479e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Sacuvani zajednicki NLL+L2: 16.409317989
Max|grad|: 2.663e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
VALIDACIJA prozor=1, L2=0.01
Srednji uslovni NLL: 16.685830857
Srednji broj pogodaka: 1.291372
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [148, 278, 212, 65, 4, 0, 0, 0]
VALIDACIJA prozor=3, L2=0.001
Trening prelaza: 3300
Broj karakteristika sa interceptom: 24
Regularizacija istorijskih koeficijenata: 0.001
Deterministicki pocetak 1/2
Sacuvani zajednicki NLL+L2: 16.107243848
Max|grad|: 8.240e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Sacuvani zajednicki NLL+L2: 16.048624389
Max|grad|: 5.234e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
VALIDACIJA prozor=3, L2=0.001
Srednji uslovni NLL: 17.335867858
Srednji broj pogodaka: 1.274399
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [148, 293, 206, 47, 10, 3, 0, 0]
VALIDACIJA prozor=3, L2=0.01
Trening prelaza: 3300
Broj karakteristika sa interceptom: 24
Regularizacija istorijskih koeficijenata: 0.01
Deterministicki pocetak 1/2
Sacuvani zajednicki NLL+L2: 16.111253709
Max|grad|: 6.594e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Sacuvani zajednicki NLL+L2: 16.045747829
Max|grad|: 4.185e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
VALIDACIJA prozor=3, L2=0.01
Srednji uslovni NLL: 17.288133928
Srednji broj pogodaka: 1.271570
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [152, 294, 190, 59, 12, 0, 0, 0]

Izabrani prozor: 1
Izabrana regularizacija: 0.01
UCENJE ZA ODVOJENI TEST
Trening prelaza: 4007
Broj karakteristika sa interceptom: 8
Regularizacija istorijskih koeficijenata: 0.01
Deterministicki pocetak 1/2
Sacuvani zajednicki NLL+L2: 16.443293866
Max|grad|: 8.493e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Sacuvani zajednicki NLL+L2: 16.433369156
Max|grad|: 6.156e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
ODVOJENI HRONOLOSKI TEST
Srednji uslovni NLL: 16.667330723
Srednji broj pogodaka: 1.223164
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [169, 289, 184, 56, 9, 1, 0, 0]
ZAVRSNO UCENJE NA SVIM PRELAZIMA
Trening prelaza: 4715
Broj karakteristika sa interceptom: 8
Regularizacija istorijskih koeficijenata: 0.01
Deterministicki pocetak 1/2
Sacuvani zajednicki NLL+L2: 16.460931000
Max|grad|: 3.974e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Sacuvani zajednicki NLL+L2: 16.450411264
Max|grad|: 4.420e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Cistoce pojedinacnih kubita:
Registar 1: 0.98324082 0.97346510 0.60269696 0.52904767 0.55442929 0.69640596
Registar 2: 0.95885338 0.66591479 0.56729963 0.61968946 0.73423928 0.86534359
Registar 3: 0.83591473 0.50074323 0.50000114 0.50037860 0.51002564 0.51106600
Registar 4: 0.90638562 0.61241702 0.50009086 0.50031268 0.91242545 0.92951604
Registar 5: 0.99947679 0.56554137 0.50001490 0.50011256 0.52937334 0.52963020
Registar 6: 0.99961675 0.74267305 0.50117791 0.50000199 0.50024606 0.80010775
Registar 7: 0.99928520 0.62793559 0.51926599 0.50059516 0.50124869 0.65861354
Log legalne mase: -14.516082326
Log uslovne mase NEXT: -14.877744590
Log odnos prvog i drugog maksimuma: 0.008004766
Promena kroz istorijske kontekste: 2.161541031e-02
Uticaj trenutne istorije na legalne marginale: 1.733158316e-02
Upozorenje: niz od 6 uzastopnih brojeva.
Raspon NEXT kombinacije: 8
Udeo istorijskih raspona <= NEXT raspon: 0.000%
Upozorenje: izrazito grupisana NEXT kombinacija.

========================================================================
loto7_4716_k89_loto_2980.csv
Stvarni broj kola: 2979
SHA256: f1cd81cd04660d50d9058675924da34346fc006ce8dabd84d52266c66afe17d1
Trening prelaza: 2084
Validacionih prelaza: 447
Test prelaza: 447
VALIDACIJA prozor=1, L2=0.001
Trening prelaza: 2084
Broj karakteristika sa interceptom: 8
Regularizacija istorijskih koeficijenata: 0.001
Deterministicki pocetak 1/2
Sacuvani zajednicki NLL+L2: 16.336980271
Max|grad|: 1.529e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Sacuvani zajednicki NLL+L2: 16.302670876
Max|grad|: 1.073e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
VALIDACIJA prozor=1, L2=0.01
Srednji uslovni NLL: 16.843976084
Srednji broj pogodaka: 1.290828
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [98, 180, 118, 43, 8, 0, 0, 0]
VALIDACIJA prozor=3, L2=0.001
Trening prelaza: 2084
Broj karakteristika sa interceptom: 24
Regularizacija istorijskih koeficijenata: 0.001
Deterministicki pocetak 1/2
Sacuvani zajednicki NLL+L2: 15.770846172
Max|grad|: 2.624e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Sacuvani zajednicki NLL+L2: 15.688795213
Max|grad|: 4.347e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
VALIDACIJA prozor=3, L2=0.001
Srednji uslovni NLL: 17.748094139
Srednji broj pogodaka: 1.255034
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [106, 168, 130, 39, 4, 0, 0, 0]
VALIDACIJA prozor=3, L2=0.01
Trening prelaza: 2084
Broj karakteristika sa interceptom: 24
Regularizacija istorijskih koeficijenata: 0.01
Deterministicki pocetak 1/2
Sacuvani zajednicki NLL+L2: 15.770756834
Max|grad|: 2.012e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Sacuvani zajednicki NLL+L2: 15.690925392
Max|grad|: 2.216e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
VALIDACIJA prozor=3, L2=0.01
Srednji uslovni NLL: 17.681458088
Srednji broj pogodaka: 1.225951
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [107, 181, 116, 37, 6, 0, 0, 0]

Izabrani prozor: 1
Izabrana regularizacija: 0.01
UCENJE ZA ODVOJENI TEST
Trening prelaza: 2531
Broj karakteristika sa interceptom: 8
Regularizacija istorijskih koeficijenata: 0.01
Deterministicki pocetak 1/2
Sacuvani zajednicki NLL+L2: 16.365092249
Max|grad|: 4.104e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Sacuvani zajednicki NLL+L2: 16.365092249
Max|grad|: 4.104e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Sacuvani zajednicki NLL+L2: 16.352684704
Max|grad|: 2.269e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
ODVOJENI HRONOLOSKI TEST
Srednji uslovni NLL: 16.840058596
Srednji broj pogodaka: 1.214765
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [107, 183, 117, 34, 6, 0, 0, 0]
ZAVRSNO UCENJE NA SVIM PRELAZIMA
Trening prelaza: 2978
Broj karakteristika sa interceptom: 8
Regularizacija istorijskih koeficijenata: 0.01
Deterministicki pocetak 1/2
Sacuvani zajednicki NLL+L2: 16.398141486
Max|grad|: 5.280e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Sacuvani zajednicki NLL+L2: 16.389339058
Max|grad|: 4.191e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Cistoce pojedinacnih kubita:
Registar 1: 0.94201332 0.84484387 0.79341551 0.70596712 0.63045669 0.73830009
Registar 2: 0.94613384 0.84093578 0.66260149 0.67863034 0.78364257 0.83826238
Registar 3: 0.99172242 0.51872375 0.50000734 0.50014218 0.59464972 0.62820725
Registar 4: 0.98763406 0.51653768 0.50014212 0.50411751 0.52168092 0.52209772
Registar 5: 0.99898850 0.67019404 0.50004356 0.50010614 0.79309711 0.84645786
Registar 6: 0.96844443 0.92493873 0.50076699 0.50001458 0.50371657 0.95011428
Registar 7: 0.98349969 0.89581486 0.59025487 0.50000282 0.50000156 0.58285576
Log legalne mase: -15.004034195
Log uslovne mase NEXT: -14.233015509
Log odnos prvog i drugog maksimuma: 0.044612929
Promena kroz istorijske kontekste: 3.278597486e-02
Uticaj trenutne istorije na legalne marginale: 2.821164432e-02
Raspon NEXT kombinacije: 25
Udeo istorijskih raspona <= NEXT raspon: 20.443%

========================================================================
loto7_4716_k89_loto_plus_1736.csv
Stvarni broj kola: 1735
SHA256: 56225ed0f0be7e2d3e30779804d8fa45e96ecfb9d14441eb2a731dd7fff38ae6
Trening prelaza: 1213
Validacionih prelaza: 260
Test prelaza: 261
VALIDACIJA prozor=1, L2=0.001
Trening prelaza: 1213
Broj karakteristika sa interceptom: 8
Regularizacija istorijskih koeficijenata: 0.001
Deterministicki pocetak 1/2
Sacuvani zajednicki NLL+L2: 16.142695412
Max|grad|: 3.945e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Sacuvani zajednicki NLL+L2: 16.064397227
Max|grad|: 8.053e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
VALIDACIJA prozor=1, L2=0.001
Srednji uslovni NLL: 17.305609796
Srednji broj pogodaka: 1.226923
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [53, 117, 70, 18, 2, 0, 0, 0]
VALIDACIJA prozor=1, L2=0.01
Trening prelaza: 1213
Broj karakteristika sa interceptom: 8
Regularizacija istorijskih koeficijenata: 0.01
Deterministicki pocetak 1/2
Sacuvani zajednicki NLL+L2: 16.134344042
Max|grad|: 2.121e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Sacuvani zajednicki NLL+L2: 16.069205750
Max|grad|: 7.785e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
VALIDACIJA prozor=1, L2=0.01
Srednji uslovni NLL: 17.328983353
Srednji broj pogodaka: 1.226923
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [58, 108, 74, 17, 3, 0, 0, 0]
VALIDACIJA prozor=3, L2=0.001
Trening prelaza: 1213
Broj karakteristika sa interceptom: 24
Regularizacija istorijskih koeficijenata: 0.001
Deterministicki pocetak 1/2
Sacuvani zajednicki NLL+L2: 15.007256304
Max|grad|: 2.952e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Sacuvani zajednicki NLL+L2: 14.787990998
Max|grad|: 1.816e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
VALIDACIJA prozor=3, L2=0.001
Srednji uslovni NLL: 19.560742691
Srednji broj pogodaka: 1.288462
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [56, 105, 70, 26, 3, 0, 0, 0]
VALIDACIJA prozor=3, L2=0.01
Trening prelaza: 1213
Broj karakteristika sa interceptom: 24
Regularizacija istorijskih koeficijenata: 0.01
Deterministicki pocetak 1/2
Sacuvani zajednicki NLL+L2: 15.006829508
Max|grad|: 1.102e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Sacuvani zajednicki NLL+L2: 14.784730931
Max|grad|: 1.236e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
VALIDACIJA prozor=3, L2=0.01
Srednji uslovni NLL: 19.478437179
Srednji broj pogodaka: 1.269231
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [57, 103, 76, 21, 3, 0, 0, 0]

Izabrani prozor: 1
Izabrana regularizacija: 0.001
UCENJE ZA ODVOJENI TEST
Trening prelaza: 1473
Broj karakteristika sa interceptom: 8
Regularizacija istorijskih koeficijenata: 0.001
Deterministicki pocetak 1/2
Sacuvani zajednicki NLL+L2: 16.214005262
Max|grad|: 3.435e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Sacuvani zajednicki NLL+L2: 16.158058736
Max|grad|: 5.516e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
ODVOJENI HRONOLOSKI TEST
Srednji uslovni NLL: 16.982103580
Srednji broj pogodaka: 1.229885
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [61, 101, 78, 20, 1, 0, 0, 0]
ZAVRSNO UCENJE NA SVIM PRELAZIMA
Trening prelaza: 1734
Broj karakteristika sa interceptom: 8
Regularizacija istorijskih koeficijenata: 0.001
Deterministicki pocetak 1/2
Sacuvani zajednicki NLL+L2: 16.270152268
Max|grad|: 4.990e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Sacuvani zajednicki NLL+L2: 16.217336044
Max|grad|: 1.863e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali gradijent nije potvrdjen.
Cistoce pojedinacnih kubita:
Registar 1: 0.99214834 0.97313520 0.64699035 0.63868817 0.60399464 0.61464920
Registar 2: 0.98816319 0.70668860 0.50080404 0.50004366 0.50163750 0.55956495
Registar 3: 0.73356312 0.51489250 0.50027076 0.50513417 0.56078897 0.56614289
Registar 4: 0.96257313 0.66298904 0.50022547 0.50063953 0.51120127 0.51120582
Registar 5: 0.99998404 0.98512091 0.54087785 0.53967244 0.92465409 0.95096224
Registar 6: 0.87349475 0.85355995 0.53916213 0.50173288 0.51865173 0.94585690
Registar 7: 0.97548791 0.70066560 0.65154753 0.51550983 0.50006977 0.50161549
Log legalne mase: -14.127130153
Log uslovne mase NEXT: -13.489275112
Log odnos prvog i drugog maksimuma: 0.040854891
Promena kroz istorijske kontekste: 6.150533676e-02
Uticaj trenutne istorije na legalne marginale: 4.626466676e-02
Raspon NEXT kombinacije: 12
Udeo istorijskih raspona <= NEXT raspon: 0.346%
Upozorenje: izrazito grupisana NEXT kombinacija.

========================================================================
POREDJENJE NEZAVISNIH MODELA

loto7_4716_k89.csv
loto7_4716_k89_loto_2980.csv
Najveca razlika sirovih raspodela: 3.791657304e-02
Najveca razlika legalnih marginala: 3.762847078e-02
Zajednickih NEXT brojeva: 1

loto7_4716_k89.csv
loto7_4716_k89_loto_plus_1736.csv
Najveca razlika sirovih raspodela: 4.510974764e-02
Najveca razlika legalnih marginala: 4.313941990e-02
Zajednickih NEXT brojeva: 0

loto7_4716_k89_loto_2980.csv
loto7_4716_k89_loto_plus_1736.csv
Najveca razlika sirovih raspodela: 4.125993824e-02
Najveca razlika legalnih marginala: 4.276602758e-02
Zajednickih NEXT brojeva: 1


========================================================================

loto7_4716_k89.csv NEXT: 7 8 9 10 11 12 15

loto7_4716_k89_loto_2980.csv NEXT: 1 3 6 7 13 23 26

loto7_4716_k89_loto_plus_1736.csv NEXT: 26 29 31 32 33 37 38
"""



"""
V1 već daje različite raspodele za prva dva CSV-a, ali isti maksimum. 
Sačuvani ispis pokazuje razliku sirovih verovatnoća oko 0.0101.

Quantum_Entanglement_V2.py zadržava isto kvantno kolo. 
Menja obuku tako da neposredno uči zajedničku raspodelu legalnih kombinacija. 
Uklanja kaznu nad osnovnim uglovima, standardizuje istorijske ulaze samo na treningu i koristi dva deterministička početka optimizacije.
"""
