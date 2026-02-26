module.exports = {
  apps: [
    {
      name: 'airehealth-ai-service',
      script: 'uvicorn',
      args: 'app.main:app --host 0.0.0.0 --port 8000',
      interpreter: 'none',
      cwd: '/home/umair/projects/airehealth-ai-service',
      env: {
        DB_HOST: 'localhost',
        DB_PORT: 3306,
        DB_USER: 'root',
        DB_PASSWORD: '',
        DB_NAME: '',
        OPENAI_API_KEY: '',
        HOST: '0.0.0.0',
        PORT: 8000,
        LOG_LEVEL: 'info',
        ENVIRONMENT: 'production',
      },
    },
  ],
};
