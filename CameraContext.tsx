// frontend/src/contexts/CameraContext.tsx
import React, { createContext, useContext, useRef, useState } from "react";

type CameraContextType = {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  capturedImage: string | null;
  setCapturedImage: (s: string | null) => void;
  isLoading: boolean;
  setIsLoading: (b: boolean) => void;
  results: any[]; // matches from backend
  setResults: (r: any[]) => void;
};

const CameraContext = createContext<CameraContextType | undefined>(undefined);

export const CameraProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [results, setResults] = useState<any[]>([]);

  const value: CameraContextType = {
    videoRef,
    canvasRef,
    capturedImage,
    setCapturedImage,
    isLoading,
    setIsLoading,
    results,
    setResults,
  };

  return <CameraContext.Provider value={value}>{children}</CameraContext.Provider>;
};

export function useCamera() {
  const ctx = useContext(CameraContext);
  if (!ctx) throw new Error("useCamera must be used inside CameraProvider");
  return ctx;
}
