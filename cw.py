import numpy as np
from tensorflow.keras.models import load_model
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import CarliniL2Method
import tensorflow as tf
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Load data and model
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

model = load_model("lstm_jammer_detection.h5")

# Wrap the model for ART
classifier = KerasClassifier(model=model, clip_values=(0, 1))

# Apply Carlini & Wagner Attack
cw_attack = CarliniL2Method(classifier=classifier, confidence=0.5, max_iter=100, learning_rate=0.01)
X_test_adv = cw_attack.generate(X_test)

# Evaluate on adversarial examples
adv_accuracy = np.sum((model.predict(X_test_adv) > 0.5).astype("int32").flatten() == y_test) / y_test.shape[0]
print(f"Accuracy on adversarial examples: {adv_accuracy:.2f}")

# Save adversarial examples
np.save("X_test_adv.npy", X_test_adv)

# Classification report
y_pred_adv = (model.predict(X_test_adv) > 0.5).astype("int32")
print("Classification Report on Adversarial Test Data:")
print(classification_report(y_test, y_pred_adv))

# Visualize an adversarial example
index = 0  # Pick a test sample
plt.plot(X_test[index].flatten(), label="Original")
plt.plot(X_test_adv[index].flatten(), label="Adversarial")
plt.legend()
plt.title("Original vs Adversarial Example")
plt.show()

