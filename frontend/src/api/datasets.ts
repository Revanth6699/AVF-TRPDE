import { apiClient } from "./client";
import type {
  DatasetListResponse,
  DatasetRequest,
  DatasetResponse,
} from "../types/api";

export async function createDataset(
  payload: DatasetRequest,
): Promise<DatasetResponse> {
  const response = await apiClient.post<DatasetResponse>(
    "/datasets",
    payload,
  );

  return response.data;
}

export async function validateDataset(
  payload: DatasetRequest,
): Promise<DatasetResponse> {
  const response = await apiClient.post<DatasetResponse>(
    "/datasets/validate",
    payload,
  );

  return response.data;
}

export async function getDatasets(): Promise<DatasetListResponse> {
  const response =
    await apiClient.get<DatasetListResponse>("/datasets");

  return response.data;
}