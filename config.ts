// Backend ka base URL
// Agar frontend aur backend same host:port pe run ho rahe hain (Vite dev:3000 & Flask:5000), 
// to yahan Flask ka address daal do
export const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:5000";
