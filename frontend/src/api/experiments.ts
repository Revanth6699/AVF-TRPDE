import { apiClient } from "./client";

export interface ExperimentRequest {
  name: string;
  description: string;
  models: string[];
  metrics: string[];
  risk_measures: string[];
  parameters: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export interface ExperimentResponse extends ExperimentRequest {
  status: string;
}

export interface ExperimentListResponse {
  experiments: ExperimentResponse[];
  count: number;
}

export interface ExperimentRunRequest {
  experiment_names: string[];
}

export interface ExperimentRunResult {
  name: string;
  status: string;
  result: Record<string, unknown>;
  error: string | null;
}

export interface ExperimentRunResponse {
  experiment_count: number;
  successful_count: number;
  failed_count: number;
  all_successful: boolean;
  results: ExperimentRunResult[];
}

export async function createExperiment(
  payload: ExperimentRequest,
): Promise<ExperimentResponse> {
  const response =
    await apiClient.post<ExperimentResponse>(
      "/experiments",
      payload,
    );

  return response.data;
}

export async function getExperiments(): Promise<ExperimentListResponse> {
  const response =
    await apiClient.get<ExperimentListResponse>(
      "/experiments",
    );

  return response.data;
}

export async function getExperiment(
  name: string,
): Promise<ExperimentResponse> {
  const response =
    await apiClient.get<ExperimentResponse>(
      `/experiments/${encodeURIComponent(name)}`,
    );

  return response.data;
}

export async function runExperiments(
  payload: ExperimentRunRequest,
): Promise<ExperimentRunResponse> {
  const response =
    await apiClient.post<ExperimentRunResponse>(
      "/experiments/run",
      payload,
    );

  return response.data;
}