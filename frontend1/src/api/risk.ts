import { apiClient } from "./client";

export interface RiskRequest {
  experiment_name: string;
  dataset_name: string;
  model: string;
  assets: string[];
  confidence_levels: number[];
  start_timestamp: string;
  end_timestamp: string;
}

export interface RiskPoint {
  timestamp: string;
  asset_id: string;
  var_95: number;
  var_99: number;
  es_95: number;
}

export interface RiskResponse {
  experiment_name: string;
  dataset_name: string;
  model: string;
  risk_points: RiskPoint[];
  observation_count: number;
  asset_count: number;
  status: string;
}

export interface RiskBacktestResult {
  model: string;
  confidence_level: number;
  measure: string;
  observations: number;
  violations: number;
  violation_rate: number;
  statistic: number | null;
  p_value: number | null;
}

export interface RiskBacktestResponse {
  experiment_name: string;
  results: RiskBacktestResult[];
  observation_count: number;
  asset_count: number;
  warnings?: string[];
}

export interface RiskComparisonItem {
  model: string;
  var_95: number;
  var_99: number;
  es_95: number;
}

export interface RiskComparisonResponse {
  experiment_name: string;
  models: RiskComparisonItem[];
  observation_count: number;
  asset_count: number;
}

export async function createRiskAnalysis(payload: RiskRequest): Promise<RiskResponse> {
  const response = await apiClient.post<RiskResponse>("/risk", payload);
  return response.data;
}

export async function getRiskAnalysis(experimentName: string, model?: string): Promise<RiskResponse> {
  const response = await apiClient.get<RiskResponse>(
    `/risk/${encodeURIComponent(experimentName)}`,
    { params: model ? { model } : undefined },
  );
  return response.data;
}

export async function backtestRisk(payload: RiskRequest): Promise<RiskBacktestResponse> {
  const response = await apiClient.post<RiskBacktestResponse>("/risk/backtest", payload);
  return response.data;
}

export async function compareRisk(payload: RiskRequest): Promise<RiskComparisonResponse> {
  const response = await apiClient.post<RiskComparisonResponse>("/risk/compare", payload);
  return response.data;
}
