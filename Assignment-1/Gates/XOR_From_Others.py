import numpy as np

# -----------------------------
# INPUT DATA
# -----------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target Outputs
y1_OR = np.array([0, 1, 1, 1])
y2_NAND = np.array([1, 1, 1, 0])
y3_XOR = np.array([0, 1, 1, 0])

learning_rate = 0.5


# -----------------------------
# PERCEPTRON TRAINING (ONE PASS)
# -----------------------------
def train_perceptron(X, y, w1, w2, threshold, title):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for i in range(len(X)):

        x1, x2 = X[i]

        print(f"\nInstance {i + 1}")
        print(f"Input : ({x1}, {x2})")
        print(f"Target: {y[i]}")

        # Net Calculation
        net = x1 * w1 + x2 * w2

        print(f"\nNet = ({x1} × {w1:.2f}) + ({x2} × {w2:.2f})")
        print(f"    = {net:.2f}")

        # Prediction
        if net >= threshold:
            prediction = 1
        else:
            prediction = 0

        print(f"Threshold  = {threshold}")
        print(f"Prediction = {prediction}")

        # Error
        error = y[i] - prediction
        print(f"Error = {y[i]} - {prediction} = {error}")

        # Weight Update
        if error != 0:

            old_w1 = w1
            old_w2 = w2

            delta_w1 = learning_rate * error * x1
            delta_w2 = learning_rate * error * x2

            w1 += delta_w1
            w2 += delta_w2

            print("\nWeight Update:")
            print(f"Δw1 = {learning_rate} × {error} × {x1} = {delta_w1:.2f}")
            print(f"Δw2 = {learning_rate} × {error} × {x2} = {delta_w2:.2f}")

            print(f"New w1 = {old_w1:.2f} + ({delta_w1:.2f}) = {w1:.2f}")
            print(f"New w2 = {old_w2:.2f} + ({delta_w2:.2f}) = {w2:.2f}")

        else:
            print("No Weight Update Required.")

    print("\nFinal Weights")
    print(f"w1 = {w1:.2f}")
    print(f"w2 = {w2:.2f}")

    return w1, w2, threshold


# -----------------------------
# TRAIN OR GATE
# -----------------------------
w1_or, w2_or, th_or = train_perceptron(
    X,
    y1_OR,
    w1=1.2,
    w2=0.6,
    threshold=1,
    title="OR GATE"
)


# -----------------------------
# TRAIN NAND GATE
# -----------------------------
w1_nand, w2_nand, th_nand = train_perceptron(
    X,
    y2_NAND,
    w1=-1.2,
    w2=-0.6,
    threshold=-1,
    title="NAND GATE"
)


# -----------------------------
# HIDDEN LAYER
# -----------------------------
print("\n" + "=" * 60)
print("HIDDEN LAYER CALCULATIONS")
print("=" * 60)

z = []

for i in range(len(X)):

    x1, x2 = X[i]

    print(f"\nInput : {X[i]}")

    # OR Neuron
    net1 = x1 * w1_or + x2 * w2_or
    z1 = 1 if net1 >= th_or else 0

    print(f"\nOR Neuron")
    print(f"Net = ({x1} × {w1_or:.2f}) + ({x2} × {w2_or:.2f})")
    print(f"    = {net1:.2f}")
    print(f"Output = {z1}")

    # NAND Neuron
    net2 = x1 * w1_nand + x2 * w2_nand
    z2 = 1 if net2 >= th_nand else 0

    print(f"\nNAND Neuron")
    print(f"Net = ({x1} × {w1_nand:.2f}) + ({x2} × {w2_nand:.2f})")
    print(f"    = {net2:.2f}")
    print(f"Output = {z2}")

    z.append([z1, z2])

z = np.array(z)

print("\nHidden Layer Outputs")
print(z)


# -----------------------------
# TRAIN FINAL AND
# -----------------------------
w1_and, w2_and, th_and = train_perceptron(
    z,
    y3_XOR,
    w1=1.2,
    w2=0.6,
    threshold=1,
    title="FINAL AND GATE"
)


# -----------------------------
# TEST XOR
# -----------------------------
print("\n" + "=" * 60)
print("XOR TESTING")
print("=" * 60)

for i in range(len(z)):

    z1, z2 = z[i]

    net = z1 * w1_and + z2 * w2_and

    if net >= th_and:
        output = 1
    else:
        output = 0

    print(f"\nOriginal Input : {X[i]}")
    print(f"Hidden Output  : ({z1}, {z2})")

    print(f"\nFinal AND Calculation")
    print(f"Net = ({z1} × {w1_and:.2f}) + ({z2} × {w2_and:.2f})")
    print(f"    = {net:.2f}")

    print(f"Prediction = {output}")
    print(f"Actual     = {y3_XOR[i]}")