import { apiClient } from "./client";

export interface StressRunRequest {
  experiment_name: string;
  strategy: string;
}

export interface StressScenarioResult {
  [key: string]: string | number | boolean | null | undefined;
}

export interface StressRunResponse {
  experiment_name: string;
  strategy: string;
  status: string;
  scenario_count: number;
  total_observations: number;
  results: StressScenarioResult[];
  summary: Record<string, unknown>;
}

export async function runStressAnalysis(payload: StressRunRequest): Promise<StressRunResponse> {
  const response = await apiClient.post<StressRunResponse>("/stress/run", payload);
  return response.data;
}

export async function getStressResults(
  experimentName: string,
  strategy?: string,
): Promise<StressRunResponse[]> {
  const response = await apiClient.get<StressRunResponse[]>(
    `/stress/${encodeURIComponent(experimentName)}`,
    { params: strategy ? { strategy } : undefined },
  );
  return response.data;
}
