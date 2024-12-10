import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy

# Load the trained model and test data
model = load_model("lstm_jammer_detection.h5")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Convert labels to one-hot if needed
y_test_one_hot = y_test

# Define the C&W attack
def carlini_wagner_attack(model, X, y, confidence=0.5, max_iter=1000, learning_rate=0.01):
    """
    Carlini & Wagner attack implementation.
    
    Parameters:
        model: Trained Keras model
        X: Input samples
        y: True labels (or target labels for targeted attack)
        confidence: Confidence of adversarial examples
        max_iter: Maximum number of iterations
        learning_rate: Learning rate for gradient descent
    
    Returns:
        Adversarial examples
    """
    # Convert input to TensorFlow tensor
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    
    # Initialize perturbation
    delta = tf.Variable(tf.zeros_like(X))
    
    # Define loss
    bce = BinaryCrossentropy()
    
    # Optimizer
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    
    # Perform optimization
    for step in range(max_iter):
        with tf.GradientTape() as tape:
            # Calculate perturbed input
            X_adv = tf.clip_by_value(X + delta, 0, 1)
            
            # Calculate model predictions
            logits = model(X_adv)
            
            # Calculate loss (maximize confidence)
            loss = bce(y, logits) + confidence * tf.reduce_sum(tf.square(delta))
        
        # Compute gradients
        gradients = tape.gradient(loss, delta)
        
        # Update perturbation
        optimizer.apply_gradients([(gradients, delta)])
        
        # Print progress
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.numpy()}")

    # Return adversarial examples
    X_adv = tf.clip_by_value(X + delta, 0, 1)
    return X_adv.numpy()

# Apply the attack
print("Generating adversarial examples...")
X_test_adv = carlini_wagner_attack(model, X_test, y_test_one_hot)

# Evaluate the attack
y_pred_adv = (model.predict(X_test_adv) > 0.5).astype("int32")
accuracy_adv = np.mean(y_pred_adv.flatten() == y_test)
print(f"Accuracy on adversarial examples: {accuracy_adv:.2f}")

# Save adversarial examples
np.save("X_test_adv.npy", X_test_adv)

# Visualize one example
import matplotlib.pyplot as plt

index = 0
plt.plot(X_test[index].flatten(), label="Original")
plt.plot(X_test_adv[index].flatten(), label="Adversarial")
plt.legend()
plt.title("Original vs Adversarial Example")
plt.show()
