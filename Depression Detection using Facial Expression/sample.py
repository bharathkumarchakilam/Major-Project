from tensorflow.keras.models import load_model

# lstm_model = load_model("models/depression_lstm_model.h5")
# ffl_model = load_model("models/depression_detection_ffl.h5")
# hybrid_model = load_model("models/depression_detection_hybrid_model.h5")

# print("LSTM Input Shape:", lstm_model.input_shape)
# print("FFL Input Shape:", ffl_model.input_shape)
# print("Hybrid Input Shape:", hybrid_model.input_shape)

try:
    model = load_model('models/depression_detection_hybrid_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")