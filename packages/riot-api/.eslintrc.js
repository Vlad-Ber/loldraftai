/** @type {import("eslint").Linter.Config} */
module.exports = {
  root: true,
  extends: ["@draftking/eslint-config/library"],
  parser: "@typescript-eslint/parser",
  parserOptions: {
    project: true,
  },
};
