import axios from "axios";

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim() || "http://localhost:8000";
const API_BASE_URL = configuredBaseUrl.replace(/\/+$/, "");

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
  // Model fitting and FHS simulation can legitimately take longer than a normal page request.
  timeout: 120000,
});
