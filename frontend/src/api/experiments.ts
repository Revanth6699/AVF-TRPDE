import { apiClient } from "./client";
import type {
  ExperimentListResponse,
  ExperimentRequest,
  ExperimentResponse,
  ExperimentRunRequest,
  ExperimentRunResponse,
} from "../types/api";

export async function createExperiment(
  payload: ExperimentRequest,
): Promise<ExperimentResponse> {
  const response = await apiClient.post<ExperimentResponse>(
    "/experiments",
    payload,
  );

  return response.data;
}

export async function getExperiments(): Promise<ExperimentListResponse> {
  const response =
    await apiClient.get<ExperimentListResponse>("/experiments");

  return response.data;
}

export async function getExperiment(
  experimentName: string,
): Promise<ExperimentResponse> {
  const response = await apiClient.get<ExperimentResponse>(
    `/experiments/${encodeURIComponent(experimentName)}`,
  );

  return response.data;
}

export async function runExperiments(
  payload: ExperimentRunRequest,
): Promise<ExperimentRunResponse> {
  const response = await apiClient.post<ExperimentRunResponse>(
    "/experiments/run",
    payload,
  );

  return response.data;
}