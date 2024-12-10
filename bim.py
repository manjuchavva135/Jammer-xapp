import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy

# Load the trained model and test data
model = load_model("lstm_jammer_detection.h5")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Define the BIM attack
def basic_iterative_method(model, X, y, epsilon=0.01, alpha=0.001, max_iter=20):
    """
    Basic Iterative Method (BIM) attack implementation.
    
    Parameters:
        model: Trained Keras model
        X: Input samples
        y: True labels
        epsilon: Maximum allowable perturbation
        alpha: Step size for each iteration
        max_iter: Number of iterations
    
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
    
    # Perform BIM iterations
    for step in range(max_iter):
        with tf.GradientTape() as tape:
            # Calculate perturbed input
            X_adv = tf.clip_by_value(X + delta, 0, 1)
            
            # Calculate model predictions
            logits = model(X_adv)
            
            # Calculate loss
            loss = bce(y, logits)
        
        # Compute gradients
        gradients = tape.gradient(loss, delta)
        
        # Update perturbation
        delta.assign_add(alpha * tf.sign(gradients))
        
        # Clip perturbation to stay within epsilon
        delta.assign(tf.clip_by_value(delta, -epsilon, epsilon))
        
        # Print progress
        if step % 5 == 0:
            print(f"Step {step}, Loss: {loss.numpy()}")

    # Return adversarial examples
    X_adv = tf.clip_by_value(X + delta, 0, 1)
    return X_adv.numpy()

# Apply the attack
print("Generating adversarial examples...")
X_test_adv = basic_iterative_method(model, X_test, y_test, epsilon=0.1, alpha=0.020, max_iter=30)

# Evaluate the attack
y_pred_adv = (model.predict(X_test_adv) > 0.5).astype("int32")
accuracy_adv = np.mean(y_pred_adv.flatten() == y_test)
print(f"Accuracy on adversarial examples: {accuracy_adv:.2f}")

# Save adversarial examples
np.save("X_test_advbim.npy", X_test_adv)

# Visualize one example
import matplotlib.pyplot as plt

index = 0
plt.plot(X_test[index].flatten(), label="Original")
plt.plot(X_test_adv[index].flatten(), label="Adversarial")
plt.legend()
plt.title("Original vs Adversarial Example")
plt.savefig("original_vs_adversarial_bim.png")
plt.show()
