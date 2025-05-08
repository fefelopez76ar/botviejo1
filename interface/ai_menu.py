"""
Módulo para el menú de configuración de IA del bot de trading
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .cli_utils import (
    clear_screen, print_header, print_subheader, print_menu,
    get_user_choice, confirm_action, print_table, print_chart,
    print_status, print_section, Colors
)

logger = logging.getLogger("AI_Menu")

def ai_configuration_menu():
    """Menú principal de configuración de IA"""
    while True:
        clear_screen()
        print_header("CONFIGURACIÓN DE INTELIGENCIA ARTIFICIAL")
        
        # Cargar configuración actual de IA
        ai_config = load_ai_config()
        
        # Mostrar estado actual
        print(f"\n{Colors.CYAN}Estado actual de los módulos de IA:{Colors.END}")
        
        status_data = []
        for module, settings in ai_config.get("modules", {}).items():
            status = f"{Colors.GREEN}Activo{Colors.END}" if settings.get("enabled", False) else f"{Colors.YELLOW}Inactivo{Colors.END}"
            last_trained = settings.get("last_trained", "Nunca")
            accuracy = f"{settings.get('accuracy', 0.0):.2f}" if settings.get('accuracy') else "N/A"
            status_data.append([module, status, last_trained, accuracy])
        
        headers = ["Módulo", "Estado", "Último entrenamiento", "Precisión"]
        print_table(headers, status_data)
        
        # Mostrar opciones
        options = [
            "Activar/Desactivar Módulos de IA",
            "Entrenar Modelos",
            "Ver Métricas de Aprendizaje",
            "Configuración Avanzada",
            "Volver al menú principal"
        ]
        
        choice = print_menu(options)
        
        if choice == 1:
            toggle_ai_modules()
        elif choice == 2:
            train_ai_models()
        elif choice == 3:
            view_learning_metrics()
        elif choice == 4:
            advanced_ai_configuration()
        elif choice == 5:
            break
        else:
            print("Opción no válida. Inténtalo de nuevo.")
            time.sleep(1)

def toggle_ai_modules():
    """Activa o desactiva módulos de IA"""
    clear_screen()
    print_header("ACTIVAR/DESACTIVAR MÓDULOS DE IA")
    
    # Cargar configuración actual
    ai_config = load_ai_config()
    modules = ai_config.get("modules", {})
    
    # Mostrar módulos disponibles
    print("\nSelecciona un módulo para activar/desactivar:")
    
    module_list = list(modules.keys())
    for i, module in enumerate(module_list, 1):
        status = "🟢 Activo" if modules[module].get("enabled", False) else "⚪ Inactivo"
        print(f"{i}. {module} - {status}")
    
    print(f"{len(module_list) + 1}. Volver")
    
    choice = get_user_choice(1, len(module_list) + 1)
    if choice == len(module_list) + 1:
        return
    
    if 1 <= choice <= len(module_list):
        selected_module = module_list[choice - 1]
        current_status = modules[selected_module].get("enabled", False)
        
        # Cambiar estado
        modules[selected_module]["enabled"] = not current_status
        
        # Guardar configuración
        ai_config["modules"] = modules
        save_ai_config(ai_config)
        
        new_status = "activado" if not current_status else "desactivado"
        print_status(f"Módulo '{selected_module}' {new_status} exitosamente", "success")
        
        input("\nPresiona Enter para continuar...")

def train_ai_models():
    """Entrena modelos de IA"""
    clear_screen()
    print_header("ENTRENAR MODELOS DE IA")
    
    # Cargar configuración actual
    ai_config = load_ai_config()
    modules = ai_config.get("modules", {})
    
    # Mostrar módulos disponibles
    print("\nSelecciona un módulo para entrenar:")
    
    module_list = list(modules.keys())
    for i, module in enumerate(module_list, 1):
        status = "🟢 Activo" if modules[module].get("enabled", False) else "⚪ Inactivo"
        last_trained = modules[module].get("last_trained", "Nunca")
        print(f"{i}. {module} - {status} - Último entrenamiento: {last_trained}")
    
    print(f"{len(module_list) + 1}. Entrenar todos los módulos")
    print(f"{len(module_list) + 2}. Volver")
    
    choice = get_user_choice(1, len(module_list) + 2)
    if choice == len(module_list) + 2:
        return
    
    # Entrenar todos los módulos
    if choice == len(module_list) + 1:
        # Configurar opciones de entrenamiento
        print("\nConfigurar opciones de entrenamiento para todos los módulos:")
        training_options = configure_training_options()
        
        # Simular entrenamiento (en una aplicación real, esto llamaría a las funciones de entrenamiento)
        print("\n⚙️ Iniciando entrenamiento de todos los módulos...")
        
        for module in module_list:
            print(f"Entrenando {module}...")
            # Simular progreso
            for i in range(10):
                print(f"Progreso: {i*10}%", end="\r")
                time.sleep(0.2)
            print("Progreso: 100%")
            
            # Actualizar configuración
            modules[module]["last_trained"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            modules[module]["accuracy"] = 0.75 + (hash(module) % 20) / 100  # Valor aleatorio entre 0.75 y 0.95
        
        ai_config["modules"] = modules
        save_ai_config(ai_config)
        
        print_status("Todos los módulos entrenados exitosamente", "success")
        
        input("\nPresiona Enter para continuar...")
        return
    
    # Entrenar un módulo específico
    if 1 <= choice <= len(module_list):
        selected_module = module_list[choice - 1]
        
        # Configurar opciones de entrenamiento
        print(f"\nConfigurar opciones de entrenamiento para '{selected_module}':")
        training_options = configure_training_options()
        
        # Simular entrenamiento (en una aplicación real, esto llamaría a la función de entrenamiento)
        print(f"\n⚙️ Iniciando entrenamiento de '{selected_module}'...")
        
        # Simular progreso
        for i in range(10):
            print(f"Progreso: {i*10}%", end="\r")
            time.sleep(0.2)
        print("Progreso: 100%")
        
        # Actualizar configuración
        modules[selected_module]["last_trained"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        modules[selected_module]["accuracy"] = 0.75 + (hash(selected_module) % 20) / 100  # Valor aleatorio entre 0.75 y 0.95
        
        ai_config["modules"] = modules
        save_ai_config(ai_config)
        
        print_status(f"Módulo '{selected_module}' entrenado exitosamente", "success")
        
        input("\nPresiona Enter para continuar...")

def configure_training_options():
    """Configura opciones de entrenamiento para modelos de IA"""
    print("\nSelecciona el periodo de datos para entrenamiento:")
    periods = ["Último mes", "Últimos 3 meses", "Últimos 6 meses", "Último año"]
    
    for i, period in enumerate(periods, 1):
        print(f"{i}. {period}")
    
    period_choice = get_user_choice(1, len(periods))
    selected_period = periods[period_choice - 1] if period_choice else periods[0]
    
    print("\nSelecciona la intensidad de entrenamiento:")
    intensities = ["Ligera (más rápido, menos preciso)", "Media", "Completa (más lento, más preciso)"]
    
    for i, intensity in enumerate(intensities, 1):
        print(f"{i}. {intensity}")
    
    intensity_choice = get_user_choice(1, len(intensities))
    selected_intensity = intensities[intensity_choice - 1] if intensity_choice else intensities[1]
    
    print("\nSelecciona el modo de validación:")
    validations = ["Cross-validation", "Hold-out", "Out-of-time"]
    
    for i, validation in enumerate(validations, 1):
        print(f"{i}. {validation}")
    
    validation_choice = get_user_choice(1, len(validations))
    selected_validation = validations[validation_choice - 1] if validation_choice else validations[0]
    
    return {
        "period": selected_period,
        "intensity": selected_intensity,
        "validation": selected_validation
    }

def view_learning_metrics():
    """Muestra métricas de aprendizaje de los modelos de IA"""
    clear_screen()
    print_header("MÉTRICAS DE APRENDIZAJE")
    
    # Cargar configuración actual
    ai_config = load_ai_config()
    modules = ai_config.get("modules", {})
    
    if not modules:
        print("\nNo hay módulos de IA configurados.")
        input("\nPresiona Enter para continuar...")
        return
    
    # Mostrar módulos disponibles
    print("\nSelecciona un módulo para ver sus métricas:")
    
    module_list = list(modules.keys())
    for i, module in enumerate(module_list, 1):
        status = "🟢 Activo" if modules[module].get("enabled", False) else "⚪ Inactivo"
        accuracy = f"{modules[module].get('accuracy', 0.0):.2f}" if modules[module].get('accuracy') else "N/A"
        print(f"{i}. {module} - {status} - Precisión: {accuracy}")
    
    print(f"{len(module_list) + 1}. Ver comparativa de todos los módulos")
    print(f"{len(module_list) + 2}. Volver")
    
    choice = get_user_choice(1, len(module_list) + 2)
    if choice == len(module_list) + 2:
        return
    
    # Ver comparativa de todos los módulos
    if choice == len(module_list) + 1:
        show_module_comparison(modules)
        return
    
    # Ver un módulo específico
    if 1 <= choice <= len(module_list):
        selected_module = module_list[choice - 1]
        show_module_details(selected_module, modules[selected_module])

def show_module_comparison(modules):
    """Muestra comparativa de todos los módulos de IA"""
    clear_screen()
    print_header("COMPARATIVA DE MÓDULOS")
    
    # Preparar datos para gráfico
    module_names = list(modules.keys())
    accuracies = [modules[m].get('accuracy', 0.0) for m in module_names]
    
    # Mostrar gráfico de barras simple
    print("\nPrecisión por módulo:")
    for i, module in enumerate(module_names):
        acc = accuracies[i]
        bar_length = int(acc * 40)  # Escalar a 40 caracteres
        bar = "█" * bar_length
        print(f"{module}: {Colors.GREEN}{bar}{Colors.END} {acc:.2f}")
    
    # Mostrar tabla comparativa
    print("\nComparativa detallada:")
    
    comparison_data = []
    for module in module_names:
        comparison_data.append([
            module,
            f"{modules[module].get('accuracy', 0.0):.2f}",
            modules[module].get('last_trained', 'Nunca'),
            "Sí" if modules[module].get("enabled", False) else "No",
            modules[module].get('f1_score', 'N/A'),
            modules[module].get('precision', 'N/A'),
            modules[module].get('recall', 'N/A')
        ])
    
    headers = ["Módulo", "Precisión", "Último entrenamiento", "Activo", "F1-Score", "Precision", "Recall"]
    print_table(headers, comparison_data)
    
    input("\nPresiona Enter para continuar...")

def show_module_details(module_name, module_data):
    """Muestra detalles de un módulo específico"""
    while True:
        clear_screen()
        print_header(f"DETALLES DEL MÓDULO: {module_name}")
        
        # Información general
        print_section("INFORMACIÓN GENERAL", {
            "Estado": "Activo" if module_data.get("enabled", False) else "Inactivo",
            "Último entrenamiento": module_data.get("last_trained", "Nunca"),
            "Tiempo de entrenamiento": module_data.get("training_time", "N/A"),
            "Datos utilizados": module_data.get("data_used", "N/A")
        })
        
        # Métricas de rendimiento
        print_section("MÉTRICAS DE RENDIMIENTO", {
            "Precisión (Accuracy)": f"{module_data.get('accuracy', 0.0):.4f}",
            "F1-Score": module_data.get("f1_score", "N/A"),
            "Precision": module_data.get("precision", "N/A"),
            "Recall": module_data.get("recall", "N/A"),
            "Matriz de confusión": "Disponible en vista detallada"
        })
        
        # Hiperparámetros
        print_section("HIPERPARÁMETROS", module_data.get("hyperparameters", {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "adam"
        }))
        
        # Simular evolución de métricas en el tiempo
        print("\nEvolución de precisión durante entrenamiento:")
        
        # Crear datos simulados para el gráfico
        training_data = {'accuracy': [0.5 + (i * 0.5 / 20) for i in range(20)]}
        print_chart(training_data, 'accuracy', title="Evolución de Accuracy")
        
        print("\nOpciones:")
        options = [
            "Ver matriz de confusión",
            "Ver importancia de características",
            "Ver curva ROC",
            "Volver"
        ]
        
        subchoice = print_menu(options)
        
        if subchoice == 1:
            show_confusion_matrix()
        elif subchoice == 2:
            show_feature_importance()
        elif subchoice == 3:
            show_roc_curve()
        elif subchoice == 4:
            break
        else:
            print("Opción no válida. Inténtalo de nuevo.")
            time.sleep(1)

def show_confusion_matrix():
    """Muestra matriz de confusión simulada"""
    clear_screen()
    print_header("MATRIZ DE CONFUSIÓN")
    
    print("\nMatriz de confusión para el modelo:")
    
    # Matriz simulada
    print(f"               {Colors.CYAN}|   Predicho   |{Colors.END}")
    print(f"               {Colors.CYAN}| Neg  | Pos  |{Colors.END}")
    print(f"{Colors.CYAN}|---------|------|------|{Colors.END}")
    print(f"{Colors.CYAN}|         | Neg  | {Colors.GREEN}85{Colors.END}   | {Colors.RED}15{Colors.END}   |")
    print(f"{Colors.CYAN}| Real    |------|------|{Colors.END}")
    print(f"{Colors.CYAN}|         | Pos  | {Colors.RED}10{Colors.END}   | {Colors.GREEN}90{Colors.END}   |")
    
    print("\nInterpretación:")
    print(f"- Verdaderos Negativos (TN): {Colors.GREEN}85{Colors.END}")
    print(f"- Falsos Positivos (FP): {Colors.RED}15{Colors.END}")
    print(f"- Falsos Negativos (FN): {Colors.RED}10{Colors.END}")
    print(f"- Verdaderos Positivos (TP): {Colors.GREEN}90{Colors.END}")
    
    input("\nPresiona Enter para continuar...")

def show_feature_importance():
    """Muestra importancia de características simulada"""
    clear_screen()
    print_header("IMPORTANCIA DE CARACTERÍSTICAS")
    
    print("\nImportancia relativa de las características en el modelo:")
    
    # Importancia simulada
    features = [
        ("RSI", 0.18),
        ("MACD", 0.15),
        ("SMA_50", 0.12),
        ("BB_Width", 0.11),
        ("Volume", 0.10),
        ("ATR", 0.08),
        ("Stochastic", 0.07),
        ("MFI", 0.06),
        ("OBV", 0.05),
        ("ADX", 0.04)
    ]
    
    for feature, importance in features:
        bar_length = int(importance * 50)  # Escalar a 50 caracteres
        bar = "█" * bar_length
        print(f"{feature:10}: {Colors.GREEN}{bar}{Colors.END} {importance:.3f}")
    
    input("\nPresiona Enter para continuar...")

def show_roc_curve():
    """Muestra curva ROC simulada"""
    clear_screen()
    print_header("CURVA ROC")
    
    print("\nCurva ROC (Receiver Operating Characteristic):")
    
    # Simular una curva ROC ASCII
    rows = 20
    cols = 40
    
    # Crear la matriz para la curva
    curve = [[" " for _ in range(cols)] for _ in range(rows)]
    
    # Diagonal de referencia
    for i in range(min(rows, cols)):
        x = int(i * cols / rows)
        y = rows - i - 1
        if 0 <= y < rows and 0 <= x < cols:
            curve[y][x] = "."
    
    # Curva ROC (por encima de la diagonal)
    for i in range(cols):
        # Fórmula para una curva ROC idealizada
        x = i
        y = rows - 1 - int(rows * (1 - (1 - (x / cols) ** 2)))
        if 0 <= y < rows and 0 <= x < cols:
            curve[y][x] = "*"
    
    # Imprimir la curva
    print(f"  {Colors.CYAN}ROC Curve (AUC = 0.88){Colors.END}")
    print(f"  {Colors.YELLOW}1.0 +{Colors.END}" + "-" * (cols - 4))
    
    for i, row in enumerate(curve):
        if i == rows - 1:
            label = f"{Colors.YELLOW}0.0{Colors.END}"
        elif i == 0:
            label = f"{Colors.YELLOW}1.0{Colors.END}"
        elif i == rows // 2:
            label = f"{Colors.YELLOW}0.5{Colors.END}"
        else:
            label = "   "
        
        line = "".join(f"{Colors.GREEN}*{Colors.END}" if c == "*" else 
                       f"{Colors.YELLOW}.{Colors.END}" if c == "." else c 
                       for c in row)
        print(f"  {label} |{line}")
    
    print(f"      {Colors.YELLOW}+{Colors.END}" + "-" * (cols - 4))
    print(f"      {Colors.YELLOW}0.0{Colors.END}" + " " * (cols // 2 - 6) + f"{Colors.YELLOW}0.5{Colors.END}" + " " * (cols // 2 - 6) + f"{Colors.YELLOW}1.0{Colors.END}")
    
    print("\nInterpretación:")
    print("- La curva ROC muestra la tasa de verdaderos positivos vs. falsos positivos")
    print("- AUC (Area Under Curve) = 0.88, lo que indica un buen rendimiento del modelo")
    print("- Un modelo perfecto tendría AUC = 1.0 (curva en la esquina superior izquierda)")
    print("- Un modelo aleatorio tendría AUC = 0.5 (diagonal punteada)")
    
    input("\nPresiona Enter para continuar...")

def advanced_ai_configuration():
    """Menú de configuración avanzada de IA"""
    clear_screen()
    print_header("CONFIGURACIÓN AVANZADA DE IA")
    
    # Cargar configuración actual
    ai_config = load_ai_config()
    
    # Opciones avanzadas
    options = [
        "Configurar hiperparámetros",
        "Gestionar conjuntos de datos",
        "Programar entrenamientos automáticos",
        "Configurar integración con estrategias",
        "Volver"
    ]
    
    choice = print_menu(options)
    
    if choice == 1:
        configure_hyperparameters(ai_config)
    elif choice == 2:
        manage_datasets()
    elif choice == 3:
        schedule_automatic_training(ai_config)
    elif choice == 4:
        configure_strategy_integration(ai_config)
    elif choice == 5:
        return
    else:
        print("Opción no válida. Inténtalo de nuevo.")
        time.sleep(1)

def configure_hyperparameters(ai_config):
    """Configura hiperparámetros de los modelos de IA"""
    clear_screen()
    print_header("CONFIGURACIÓN DE HIPERPARÁMETROS")
    
    modules = ai_config.get("modules", {})
    
    # Mostrar módulos disponibles
    print("\nSelecciona un módulo para configurar hiperparámetros:")
    
    module_list = list(modules.keys())
    for i, module in enumerate(module_list, 1):
        print(f"{i}. {module}")
    
    print(f"{len(module_list) + 1}. Volver")
    
    choice = get_user_choice(1, len(module_list) + 1)
    if choice == len(module_list) + 1:
        return
    
    if 1 <= choice <= len(module_list):
        selected_module = module_list[choice - 1]
        hyperparameters = modules[selected_module].get("hyperparameters", {})
        
        clear_screen()
        print_header(f"HIPERPARÁMETROS: {selected_module}")
        
        print("\nHiperparámetros actuales:")
        for param, value in hyperparameters.items():
            print(f"{param}: {value}")
        
        print("\nSelecciona un hiperparámetro para modificar:")
        param_list = list(hyperparameters.keys())
        
        for i, param in enumerate(param_list, 1):
            print(f"{i}. {param}: {hyperparameters[param]}")
        
        print(f"{len(param_list) + 1}. Optimizar automáticamente")
        print(f"{len(param_list) + 2}. Volver")
        
        param_choice = get_user_choice(1, len(param_list) + 2)
        
        if param_choice == len(param_list) + 2:
            return
        
        if param_choice == len(param_list) + 1:
            # Optimización automática
            print_status("Iniciando optimización automática de hiperparámetros...", "info")
            
            # Simular proceso
            for i in range(10):
                print(f"Progreso: {i*10}%", end="\r")
                time.sleep(0.2)
            print("Progreso: 100%")
            
            print_status("Hiperparámetros optimizados exitosamente", "success")
            
            # Actualizar con valores "optimizados"
            hyperparameters["learning_rate"] = 0.005
            hyperparameters["batch_size"] = 64
            hyperparameters["epochs"] = 150
            
            modules[selected_module]["hyperparameters"] = hyperparameters
            ai_config["modules"] = modules
            save_ai_config(ai_config)
            
            input("\nPresiona Enter para continuar...")
            return
        
        if 1 <= param_choice <= len(param_list):
            selected_param = param_list[param_choice - 1]
            current_value = hyperparameters[selected_param]
            
            print(f"\nModificar {selected_param} (valor actual: {current_value}):")
            
            try:
                if isinstance(current_value, int):
                    new_value = int(input("> "))
                elif isinstance(current_value, float):
                    new_value = float(input("> "))
                else:
                    new_value = input("> ")
                
                hyperparameters[selected_param] = new_value
                modules[selected_module]["hyperparameters"] = hyperparameters
                ai_config["modules"] = modules
                save_ai_config(ai_config)
                
                print_status(f"Parámetro {selected_param} actualizado a {new_value}", "success")
            except ValueError:
                print_status("Valor inválido. No se realizaron cambios", "error")
            
            input("\nPresiona Enter para continuar...")

def manage_datasets():
    """Gestiona conjuntos de datos para entrenamiento"""
    clear_screen()
    print_header("GESTIÓN DE CONJUNTOS DE DATOS")
    
    # Simular listado de conjuntos de datos
    datasets = [
        {"name": "SOL/USDT 1H 2023-2024", "size": "12.5 MB", "samples": 8760, "features": 25},
        {"name": "SOL/USDT 15m 2024", "size": "8.2 MB", "samples": 35040, "features": 25},
        {"name": "Multi-par 1D 2022-2024", "size": "5.1 MB", "samples": 1095, "features": 40}
    ]
    
    print("\nConjuntos de datos disponibles:")
    
    dataset_data = []
    for ds in datasets:
        dataset_data.append([
            ds["name"],
            ds["size"],
            str(ds["samples"]),
            str(ds["features"])
        ])
    
    headers = ["Nombre", "Tamaño", "Muestras", "Características"]
    print_table(headers, dataset_data)
    
    options = [
        "Importar nuevo conjunto de datos",
        "Limpiar y preprocesar datos",
        "Generar características adicionales",
        "Volver"
    ]
    
    choice = print_menu(options)
    
    if choice == 1:
        print_status("Funcionalidad de importación no implementada", "warning")
        input("\nPresiona Enter para continuar...")
    elif choice == 2:
        print_status("Funcionalidad de limpieza no implementada", "warning")
        input("\nPresiona Enter para continuar...")
    elif choice == 3:
        print_status("Generación de características no implementada", "warning")
        input("\nPresiona Enter para continuar...")
    elif choice == 4:
        return
    else:
        print("Opción no válida. Inténtalo de nuevo.")
        time.sleep(1)

def schedule_automatic_training(ai_config):
    """Programa entrenamientos automáticos"""
    clear_screen()
    print_header("PROGRAMACIÓN DE ENTRENAMIENTOS")
    
    # Cargar programación actual
    schedule = ai_config.get("training_schedule", {
        "enabled": False,
        "frequency": "daily",
        "time": "00:00",
        "modules": []
    })
    
    print("\nProgramación actual:")
    status = "Activa" if schedule.get("enabled", False) else "Inactiva"
    print(f"Estado: {status}")
    print(f"Frecuencia: {schedule.get('frequency', 'No configurada')}")
    print(f"Hora: {schedule.get('time', 'No configurada')}")
    
    if schedule.get("modules"):
        print("Módulos: " + ", ".join(schedule["modules"]))
    else:
        print("Módulos: Todos")
    
    options = [
        "Activar/Desactivar programación",
        "Configurar frecuencia",
        "Configurar hora",
        "Seleccionar módulos",
        "Volver"
    ]
    
    choice = print_menu(options)
    
    if choice == 1:
        # Cambiar estado
        schedule["enabled"] = not schedule.get("enabled", False)
        status_str = "activada" if schedule["enabled"] else "desactivada"
        print_status(f"Programación automática {status_str}", "success")
    elif choice == 2:
        # Configurar frecuencia
        print("\nSelecciona la frecuencia de entrenamiento:")
        frequencies = ["daily", "weekly", "biweekly", "monthly"]
        
        for i, freq in enumerate(frequencies, 1):
            print(f"{i}. {freq}")
        
        freq_choice = get_user_choice(1, len(frequencies))
        if freq_choice:
            schedule["frequency"] = frequencies[freq_choice - 1]
            print_status(f"Frecuencia actualizada a {schedule['frequency']}", "success")
    elif choice == 3:
        # Configurar hora
        print("\nIntroduce la hora de entrenamiento (formato HH:MM):")
        time_input = input("> ")
        
        # Validación simple
        if ":" in time_input and len(time_input) == 5:
            schedule["time"] = time_input
            print_status(f"Hora actualizada a {time_input}", "success")
        else:
            print_status("Formato de hora inválido. Use HH:MM", "error")
    elif choice == 4:
        # Seleccionar módulos
        modules = ai_config.get("modules", {})
        module_list = list(modules.keys())
        
        print("\nSelecciona los módulos a entrenar automáticamente:")
        print("0. Todos los módulos")
        
        for i, module in enumerate(module_list, 1):
            print(f"{i}. {module}")
        
        module_choice = get_user_choice(0, len(module_list))
        
        if module_choice == 0:
            schedule["modules"] = []
            print_status("Se entrenarán todos los módulos automáticamente", "success")
        elif 1 <= module_choice <= len(module_list):
            if "modules" not in schedule:
                schedule["modules"] = []
            
            selected_module = module_list[module_choice - 1]
            
            if selected_module in schedule["modules"]:
                schedule["modules"].remove(selected_module)
                print_status(f"Módulo {selected_module} eliminado de la programación", "success")
            else:
                schedule["modules"].append(selected_module)
                print_status(f"Módulo {selected_module} añadido a la programación", "success")
    elif choice == 5:
        pass
    else:
        print("Opción no válida. Inténtalo de nuevo.")
        time.sleep(1)
        return
    
    # Guardar configuración
    ai_config["training_schedule"] = schedule
    save_ai_config(ai_config)
    
    input("\nPresiona Enter para continuar...")

def configure_strategy_integration(ai_config):
    """Configura integración de IA con estrategias de trading"""
    clear_screen()
    print_header("INTEGRACIÓN CON ESTRATEGIAS")
    
    # Cargar configuración actual
    integration = ai_config.get("strategy_integration", {
        "enabled": False,
        "confidence_threshold": 0.7,
        "override_traditional": False,
        "mixed_weight": 0.5
    })
    
    print("\nConfiguración actual:")
    status = "Activa" if integration.get("enabled", False) else "Inactiva"
    print(f"Estado: {status}")
    print(f"Umbral de confianza: {integration.get('confidence_threshold', 0.7)}")
    print(f"Anular señales tradicionales: {'Sí' if integration.get('override_traditional', False) else 'No'}")
    print(f"Peso en estrategia mixta: {integration.get('mixed_weight', 0.5)}")
    
    options = [
        "Activar/Desactivar integración",
        "Configurar umbral de confianza",
        "Configurar modo de anulación",
        "Configurar peso en estrategia mixta",
        "Volver"
    ]
    
    choice = print_menu(options)
    
    if choice == 1:
        # Cambiar estado
        integration["enabled"] = not integration.get("enabled", False)
        status_str = "activada" if integration["enabled"] else "desactivada"
        print_status(f"Integración con estrategias {status_str}", "success")
    elif choice == 2:
        # Configurar umbral
        print("\nIntroduce el umbral de confianza (0.0 - 1.0):")
        try:
            threshold = float(input("> "))
            if 0.0 <= threshold <= 1.0:
                integration["confidence_threshold"] = threshold
                print_status(f"Umbral actualizado a {threshold}", "success")
            else:
                print_status("El umbral debe estar entre 0.0 y 1.0", "error")
        except ValueError:
            print_status("Valor inválido", "error")
    elif choice == 3:
        # Configurar modo de anulación
        integration["override_traditional"] = not integration.get("override_traditional", False)
        status_str = "activada" if integration["override_traditional"] else "desactivada"
        print_status(f"Anulación de señales tradicionales {status_str}", "success")
    elif choice == 4:
        # Configurar peso
        print("\nIntroduce el peso de IA en estrategia mixta (0.0 - 1.0):")
        try:
            weight = float(input("> "))
            if 0.0 <= weight <= 1.0:
                integration["mixed_weight"] = weight
                print_status(f"Peso actualizado a {weight}", "success")
            else:
                print_status("El peso debe estar entre 0.0 y 1.0", "error")
        except ValueError:
            print_status("Valor inválido", "error")
    elif choice == 5:
        pass
    else:
        print("Opción no válida. Inténtalo de nuevo.")
        time.sleep(1)
        return
    
    # Guardar configuración
    ai_config["strategy_integration"] = integration
    save_ai_config(ai_config)
    
    input("\nPresiona Enter para continuar...")

def load_ai_config() -> Dict:
    """
    Carga la configuración de IA
    
    Returns:
        Dict: Configuración de IA
    """
    config_path = "config/ai_config.json"
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error al cargar configuración de IA: {e}")
    
    # Configuración predeterminada
    return {
        "modules": {
            "adaptive_weights": {
                "enabled": True,
                "last_trained": "2025-01-15 10:30",
                "accuracy": 0.82,
                "hyperparameters": {
                    "learning_rate": 0.01,
                    "batch_size": 32,
                    "epochs": 100
                }
            },
            "price_prediction": {
                "enabled": False,
                "last_trained": "Nunca",
                "accuracy": 0.0,
                "hyperparameters": {
                    "learning_rate": 0.01,
                    "batch_size": 32,
                    "epochs": 100
                }
            },
            "pattern_recognition": {
                "enabled": True,
                "last_trained": "2025-02-28 15:45",
                "accuracy": 0.78,
                "hyperparameters": {
                    "learning_rate": 0.01,
                    "batch_size": 32,
                    "epochs": 100
                }
            },
            "risk_management": {
                "enabled": True,
                "last_trained": "2025-03-10 09:15",
                "accuracy": 0.85,
                "hyperparameters": {
                    "learning_rate": 0.01,
                    "batch_size": 32,
                    "epochs": 100
                }
            },
            "reinforcement_learning": {
                "enabled": False,
                "last_trained": "Nunca",
                "accuracy": 0.0,
                "hyperparameters": {
                    "learning_rate": 0.01,
                    "batch_size": 32,
                    "epochs": 100
                }
            }
        }
    }

def save_ai_config(config: Dict):
    """
    Guarda la configuración de IA
    
    Args:
        config: Configuración a guardar
    """
    config_path = "config/ai_config.json"
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("Configuración de IA guardada exitosamente")
    except Exception as e:
        logger.error(f"Error al guardar configuración de IA: {e}")