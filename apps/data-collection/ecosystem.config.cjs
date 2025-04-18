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
    },
    ...["EUW1", "KR", "NA1", "OC1"].flatMap((region, regionIndex) =>
      ["collectMatchIds", "fetchPuuids", "processMatches", "updateLadder"].map(
        (script) => ({
          name: `${script}-${region}`,
          // Delay each region's ladder updates by 30 minutes to stagger the load on the database
          script:
            script === "updateLadder"
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
