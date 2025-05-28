import time
from prettytable import PrettyTable

LOG_FILE = "bot.log"


def tail_log(file_path, lines=20):
    """Lee las últimas líneas de un archivo de log."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.readlines()[-lines:]
    except FileNotFoundError:
        return []


def display_logs():
    """Muestra los logs en un formato de tabla."""
    logs = tail_log(LOG_FILE)

    if not logs:
        print("[INFO] No hay datos en el archivo de log aún.")
        return

    table = PrettyTable()
    table.field_names = ["Fecha", "Nivel", "Mensaje"]

    for log in logs:
        parts = log.split(" - ", 2)
        if len(parts) == 3:
            date, level, message = parts
            table.add_row([date.strip(), level.strip(), message.strip()])

    print("\n[Últimos Logs]")
    print(table)


if __name__ == "__main__":
    print("[INFO] Visualizador de Logs iniciado. Presiona Ctrl+C para salir.")
    try:
        while True:
            display_logs()
            time.sleep(30)  # Refrescar cada 30 segundos
    except KeyboardInterrupt:
        print("\n[INFO] Visualizador de Logs detenido.")
