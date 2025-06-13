import React, { useState } from "react";
import axios from "axios";
import "./App.css"; // Import the CSS file

function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select a .wav file");

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      const response = await axios.post("http://127.0.0.1:8000/predict/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResults(response.data);
    } catch (error) {
      console.error("Upload failed:", error);
      alert("Error uploading file or server error.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h2 className="title">🎙️ Emotion Detection from Audio (.wav)</h2>
      <input className="file-input" type="file" accept=".wav" onChange={handleFileChange} />
      <button className="upload-btn" onClick={handleUpload} disabled={loading}>
        {loading ? "Analyzing..." : "Predict Emotion"}
      </button>

      {results && (
        <div className="results">
          <h3>Predictions</h3>
          <ul>
            <li><strong>SVM:</strong> {results.svm_prediction}</li>
            <li><strong>Random Forest:</strong> {results.rf_prediction}</li>
            <li><strong>CNN:</strong> {results.cnn_prediction}</li>
            <li><strong>LSTM:</strong> {results.lstm_prediction}</li>
            <li><strong>GRU:</strong> {results.gru_prediction}</li>
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
