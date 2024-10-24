// src/utils/telemetry.ts
import * as appInsights from "applicationinsights";

export class TelemetryClient {
  private static instance: TelemetryClient;
  private client: appInsights.TelemetryClient | null = null;
  private initialized = false;

  private constructor() {
    this.initialize();
  }

  private initialize() {
    if (this.initialized) return;

    try {
      if (!process.env.APPINSIGHTS_CONNECTION_STRING) {
        console.warn(
          "APPINSIGHTS_CONNECTION_STRING is not set, telemetry will be disabled"
        );
        return;
      }

      const appInsightsClient = new appInsights.TelemetryClient(
        process.env.APPINSIGHTS_CONNECTION_STRING
      );
      appInsightsClient.config.disableAppInsights = false;
      appInsightsClient.config.enableAutoCollectConsole = false;
      appInsightsClient.config.enableAutoCollectDependencies = false;
      appInsightsClient.config.enableAutoCollectExceptions = false;
      appInsightsClient.config.enableAutoCollectHeartbeat = false;
      appInsightsClient.config.enableAutoCollectPerformance = false;
      appInsightsClient.config.enableAutoCollectRequests = false;

      this.client = appInsightsClient;
      this.initialized = true;
    } catch (error) {
      console.error("Failed to initialize Application Insights:", error);
    }
  }

  public static getInstance(): TelemetryClient {
    if (!TelemetryClient.instance) {
      TelemetryClient.instance = new TelemetryClient();
    }
    return TelemetryClient.instance;
  }

  public trackEvent(
    name: string,
    properties: {
      region?: string;
      count?: number;
      tier?: string;
      error?: string;
    } = {}
  ) {
    if (!this.client) {
      console.debug(`Event not tracked (telemetry disabled): ${name}`);
      return;
    }

    // If count is provided, include it as a measurement
    const measurements = properties.count
      ? { count: properties.count }
      : undefined;

    // Remove count from properties as it's now in measurements
    const { count, ...restProperties } = properties;

    this.client.trackEvent({
      name,
      properties: restProperties,
      measurements,
    });
  }

  public trackMetric(
    name: string,
    value: number,
    properties: { region?: string } = {}
  ) {
    if (!this.client) {
      console.debug(`Metric not tracked (telemetry disabled): ${name}`);
      return;
    }

    this.client.trackMetric({
      name,
      value,
      properties,
    });
  }

  public async flush() {
    if (this.client) {
      await this.client.flush();
    }
  }
}

// Export a pre-configured instance
export const telemetry = TelemetryClient.getInstance();
