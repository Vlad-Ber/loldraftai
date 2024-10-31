module.exports = {
  apps: [
    {
      name: "extract-to-azure",
      script: "yarn tsx ./src/scripts/extractToAzure.ts",
      interpreter: "/bin/bash",
      interpreter_args: "-c",
      env: {
        PATH: "/home/azureuser/draftking-monorepo/apps/data-collection/.venv/bin:$PATH",
        VIRTUAL_ENV:
          "/home/azureuser/draftking-monorepo/apps/data-collection/.venv",
      },
      autorestart: true,
      max_restarts: 10,
      restart_delay: 4000,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: "./logs/extractToAzure_error.log",
      out_file: "./logs/extractToAzure_out.log",
    },
    // Dynamic generation of region-specific apps
    ...["EUW1", "KR", "NA1", "OC1"].flatMap((region) =>
      ["collectMatchIds", "fetchPuuids", "processMatches", "updateLadder"].map(
        (script) => ({
          name: `${script}-${region}`,
          script: `yarn tsx ./src/scripts/${script}.ts --region ${region}`,
          autorestart: true,
          max_restarts: 10,
          restart_delay: 4000,
          log_date_format: "YYYY-MM-DD HH:mm:ss",
          error_file: `./logs/${script}_${region}_error.log`,
          out_file: `./logs/${script}_${region}_out.log`,
        })
      )
    ),
  ],
};
