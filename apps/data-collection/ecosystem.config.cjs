module.exports = {
  apps: [
    {
      name: "extract-to-azure",
      script: "./extractToAzure.sh",
      interpreter: "/bin/bash",
      cwd: "/home/azureuser/draftking-monorepo/apps/data-collection",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 4000,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: "./logs/extractToAzure_error.log",
      out_file: "./logs/extractToAzure_out.log",
      // Run for 5 hours and 55 minutes (slightly less than 1/4 of the day), then stop
      cron_restart: "0 0 * * *", // Start at midnight
      env: {
        MAX_RUNTIME_MINUTES: "355", // 5 hours and 55 minutes
      },
    },
    ...["EUW1", "KR", "NA1", "OC1"].flatMap((region, regionIndex) =>
      ["collectMatchIds", "fetchPuuids", "processMatches", "updateLadder"].map(
        (script) => ({
          name: `${script}-${region}`,
          // For processMatches, start 6 hours after midnight (with 5 min buffer)
          script:
            script === "processMatches"
              ? `sleep ${
                  6 * 60 * 60 // 6 hours = 21600 seconds
                } && yarn tsx ./src/scripts/${script}.ts --region ${region}`
              : script === "updateLadder"
              ? `sleep ${
                  regionIndex * 1800
                } && yarn tsx ./src/scripts/${script}.ts --region ${region}`
              : `yarn tsx ./src/scripts/${script}.ts --region ${region}`,
          autorestart: true,
          max_restarts: 10,
          restart_delay: 4000,
          // Add these settings for processMatches scripts
          ...(script === "processMatches"
            ? {
                // Increase kill timeout to allow for longer sleep periods
                kill_timeout: 120000,
                // Add a watch option to prevent considering the process dead during sleep
                watch: false,
                // Restart at midnight to align with the 24-hour cycle
                cron_restart: "0 0 * * *",
                // Run for 17 hours and 55 minutes (with 5 min buffer before midnight)
                env: {
                  MAX_RUNTIME_MINUTES: "1075", // 17 hours and 55 minutes (1075 minutes)
                },
              }
            : {}),
          log_date_format: "YYYY-MM-DD HH:mm:ss",
          error_file: `./logs/${script}_${region}_error.log`,
          out_file: `./logs/${script}_${region}_out.log`,
        })
      )
    ),
  ],
};
