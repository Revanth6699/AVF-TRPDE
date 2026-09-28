import { apiClient } from "./client";

export interface ForecastPoint {
  timestamp: string;
  asset_id: string;
  actual_volatility: number | null;
  predicted_volatility: number;
}

export interface ForecastResponse {
  experiment_name: string;
  dataset_name: string;
  model: string;
  forecasts: ForecastPoint[];
  observation_count: number;
  asset_count: number;
  status: string;
}

export interface ForecastRequest {
  experiment_name: string;
  dataset_name: string;
  model: string;
  assets: string[];
  start_timestamp: string;
  end_timestamp: string;
}

export interface ForecastMetric {
  model: string;
  metric: string;
  value: number;
}

export interface ForecastEvaluationResponse {
  experiment_name: string;
  metrics: ForecastMetric[];
  observation_count: number;
  asset_count: number;
}

export interface ForecastComparisonItem {
  model: string;
  mae: number;
  rmse: number;
  qlike: number;
}

export interface ForecastComparisonResponse {
  experiment_name: string;
  models: ForecastComparisonItem[];
  observation_count: number;
  asset_count: number;
}

export async function createForecast(
  payload: ForecastRequest,
): Promise<ForecastResponse> {
  const response = await apiClient.post<ForecastResponse>(
    "/forecasts",
    payload,
  );

  return response.data;
}

export async function getForecast(
  experimentName: string,
  model?: string,
): Promise<ForecastResponse> {
  const response = await apiClient.get<ForecastResponse>(
    `/forecasts/${encodeURIComponent(experimentName)}`,
    {
      params: model ? { model } : undefined,
    },
  );

  return response.data;
}

export async function evaluateForecast(
  payload: ForecastRequest,
): Promise<ForecastEvaluationResponse> {
  const response = await apiClient.post<ForecastEvaluationResponse>(
    "/forecasts/evaluate",
    payload,
  );

  return response.data;
}

export async function compareForecasts(
  payload: ForecastRequest,
): Promise<ForecastComparisonResponse> {
  const response = await apiClient.post<ForecastComparisonResponse>(
    "/forecasts/compare",
    payload,
  );

  return response.data;
}