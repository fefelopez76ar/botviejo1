# Pendientes del Proyecto

## 2025-05-30
### Problemas Detectados
1. **Error en `verificar_db.py`:**
   - Mensaje: `sqlite3.OperationalError: no such table: tickers`.
   - Acción requerida: Crear la tabla `tickers` en la base de datos `market_data.db` o verificar si la base de datos está configurada correctamente.

### Próximas Tareas
1. Revisar el esquema de la base de datos `market_data.db`.
2. Implementar un script para inicializar las tablas necesarias si no existen.
3. Validar que las tablas `tickers` y `order_book` contengan datos válidos.
4. Probar nuevamente el script `verificar_db.py` después de solucionar el problema.

### Notas Adicionales
- Asegurarse de que la base de datos esté en la ubicación correcta y accesible.
- Documentar cualquier cambio realizado en el esquema de la base de datos en `info/roadmap_features.md`.

---

## 2025-06-05
### Tareas Generales
1. **Final Testing:**
   - Run the bot to confirm that it initializes correctly, processes data, and saves it to the database without errors.
   - Verify that the database (`market_data.db`) contains the expected data using `verificar_db.py`.

2. **Error Handling:**
   - Ensure robust error handling for any remaining edge cases during bot execution.

3. **Folder and File Verification:**
   - Confirm that the `info/` folder exists and contains the necessary logs.
   - Verify that the new `market_data.db` file is created with the correct schema.

4. **Data Validation:**
   - Compress and share the `info/` folder for further analysis of saved data.

---

## Próximos Pasos
1. **Diagnóstico del Problema en VSC:** Necesito el `traceback` completo o el mensaje de error específico de VSC para identificar la causa raíz.
2. **Integración de Estrategias:** Una vez resuelto el entorno, integrar la llamada a las funciones de `ScalpingStrategies` utilizando los datos de `candle1m` y `tickers`.
3. **Manejo de la Restricción VIP:** Decidir cómo adaptar o si modificar la función `determine_liquidity_zones` dado que el Order Book TBT no está disponible.
