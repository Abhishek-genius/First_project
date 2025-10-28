import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css"; 
import "keen-slider/keen-slider.min.css";  // 👈 yaha hona chahiye

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
