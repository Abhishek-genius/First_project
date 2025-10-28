// src/App.tsx
import React from "react";
import { CameraProvider } from "./contexts/CameraContext";
import Header from "./components/Header";
import CombinedCaptureCard from "./components/CombinedCaptureCard";
import ResultsSection from "./components/ResultsSection";

function App() {
  return (
    <CameraProvider>
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        <div className="relative z-10">
          <Header />
          <main className="container mx-auto px-4 py-8 max-w-7xl">
            <div className="grid grid-cols-1 gap-8 mb-8">
              {/* single combined control card */}
              <CombinedCaptureCard />
            </div>
            <ResultsSection />
          </main>
        </div>
      </div>
    </CameraProvider>
  );
}

export default App;
