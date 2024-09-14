import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import prettier from "eslint-config-prettier";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  prettier,
  {
    ignores: ["node_modules/"],
  },
  {
    // TODO: Fix tsconfig setting in the monorepo.
    ignores: ["tests/*", "*.config.*"],
    rules: {
      "@typescript-eslint/no-unnecessary-condition": "error",
      "@typescript-eslint/require-await": "error",
      "@typescript-eslint/await-thenable": "error",
    },
    languageOptions: {
      parserOptions: {
        project: "./tsconfig.json",
      },
    },
  },
  {
    rules: {
      curly: ["error", "multi-line", "consistent"],
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["../"],
              message:
                "Relative imports from parent directories are not allowed.",
            },
          ],
        },
      ],
    },
  }
);
