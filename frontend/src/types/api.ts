export interface DatasetRequest {
  name: string;
  source: string;
  frequency: string;
  assets: string[];
  start_timestamp: string;
  end_timestamp: string;
  row_count: number;
  fingerprint: string;
}

export interface DatasetResponse extends DatasetRequest {
  status: string;
}

export interface DatasetListResponse {
  datasets: DatasetResponse[];
  count: number;
}

export interface ExperimentRequest {
  name: string;
  description?: string;
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

export interface ExperimentRunResponse {
  experiment_count: number;
  successful_count: number;
  failed_count: number;
  all_successful: boolean;
  results: Record<string, unknown>[];
}

export interface ForecastRequest {
  experiment_name: string;
  dataset_name: string;
  model: string;
  assets: string[];
  start_timestamp: string;
  end_timestamp: string;
}

export interface ForecastPoint {
  timestamp: string;
  asset_id: string;
  actual_volatility?: number | null;
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
  statistic?: number | null;
  p_value?: number | null;
}

export interface RiskBacktestResponse {
  experiment_name: string;
  results: RiskBacktestResult[];
  observation_count: number;
  asset_count: number;
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
  metrics: Record<string, number | string | null>;
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
  error?: string | null;
}

export interface ReportRunResponse {
  experiment_name: string;
  status: string;
  reports: ReportGenerationResult[];
  conclusion?: ResearchConclusion | null;
}