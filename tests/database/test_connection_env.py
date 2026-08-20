from src.database.connection import DatabaseSettings, load_dotenv


def test_settings_read_db_star_from_dotenv(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DB_HOST=localhost\nDB_PORT=5432\nDB_NAME=student_db\nDB_USER=postgres\nDB_PASSWORD=secret\n",
        encoding="utf-8",
    )
    for key in (
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
        "POSTGRES_HOST",
        "DATABASE_URL",
        "FINAL_DATABASE_URL",
        "POSTGRES_RUNTIME_APP_DSN",
    ):
        monkeypatch.delenv(key, raising=False)
    load_dotenv(env_file)
    settings = DatabaseSettings.from_environment()
    assert settings.host == "localhost"
    assert settings.database == "student_db"
    assert settings.user == "postgres"
    assert settings.password == "secret"
