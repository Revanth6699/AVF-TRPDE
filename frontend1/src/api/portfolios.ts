import { apiClient } from "./client";

export interface PortfolioRequest {
  experiment_name: string;
  dataset_name: string;
  strategy: string;
  assets: string[];
  start_timestamp: string;
  end_timestamp: string;
  volatility_target: number;
  max_position: number;
  max_turnover: number;
  transaction_cost: number;
}

export interface PortfolioPosition {
  timestamp: string;
  asset_id: string;
  weight: number;
}

export interface PortfolioResponse {
  experiment_name: string;
  dataset_name: string;
  strategy: string;
  positions: PortfolioPosition[];
  observation_count: number;
  asset_count: number;
  status: string;
}

export interface PortfolioComparisonItem {
  strategy: string;
  total_return: number;
  volatility: number;
  sharpe: number;
  sortino: number;
  max_drawdown: number;
  calmar: number;
  var_95: number;
  var_99: number;
  es_95: number;
  turnover: number;
  transaction_costs: number;
  gross_pnl: number;
  net_pnl: number;
}

export interface PortfolioComparisonResponse {
  experiment_name: string;
  strategies: PortfolioComparisonItem[];
  observation_count: number;
  asset_count: number;
}

export async function createPortfolio(
  payload: PortfolioRequest,
): Promise<PortfolioResponse> {
  const response = await apiClient.post<PortfolioResponse>(
    "/portfolios",
    payload,
  );

  return response.data;
}

export async function getPortfolio(
  experimentName: string,
  strategy?: string,
): Promise<PortfolioResponse> {
  const response = await apiClient.get<PortfolioResponse>(
    `/portfolios/${encodeURIComponent(experimentName)}`,
    {
      params: strategy ? { strategy } : undefined,
    },
  );

  return response.data;
}

export async function comparePortfolios(
  payload: PortfolioRequest,
): Promise<PortfolioComparisonResponse> {
  const response = await apiClient.post<PortfolioComparisonResponse>(
    "/portfolios/compare",
    payload,
  );

  return response.data;
}