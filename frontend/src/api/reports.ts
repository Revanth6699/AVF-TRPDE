import { apiClient } from "./client";

export interface ReportRequest {
  experiment_name: string;
  dataset_name: string;
  report_type: string;
  start_timestamp: string;
  end_timestamp: string;
  include_forecasts: boolean;
  include_risk: boolean;
  include_portfolio: boolean;
  include_stress: boolean;
  include_statistical_tests: boolean;
}

export interface ReportSection {
  title: string;
  content: string;
  metrics: Record<string, number>;
}

export interface ReportResponse {
  experiment_name: string;
  dataset_name: string;
  report_type: string;
  generated_at: string;
  sections: ReportSection[];
  status: string;
}

export interface ResearchConclusion {
  research_question: string;
  null_hypothesis: string;
  alternative_hypothesis: string;
  conclusion: string;
  statistical_significance: boolean;
  economic_significance: boolean;
  supporting_evidence: string[];
  limitations: string[];
}

export interface ReportArtifact {
  path: string;
  artifact_type: string;
  size_bytes: number;
}

export interface ReportGenerationResult {
  report_type: string;
  status: string;
  artifacts: ReportArtifact[];
  error: string | null;
}

export interface ReportRunResponse {
  experiment_name: string;
  status: string;
  reports: ReportGenerationResult[];
  conclusion: ResearchConclusion | null;
}

export async function createReport(
  payload: ReportRequest,
): Promise<ReportResponse> {
  const response = await apiClient.post<ReportResponse>(
    "/reports",
    payload,
  );

  return response.data;
}

export async function getReport(
  experimentName: string,
  reportType?: string,
): Promise<ReportResponse> {
  const response = await apiClient.get<ReportResponse>(
    `/reports/${encodeURIComponent(experimentName)}`,
    {
      params: reportType ? { report_type: reportType } : undefined,
    },
  );

  return response.data;
}

export async function runReport(
  payload: ReportRequest,
): Promise<ReportRunResponse> {
  const response = await apiClient.post<ReportRunResponse>(
    "/reports/run",
    payload,
  );

  return response.data;
}