# Limpieza solicitada: datasets anteriores y sus resultados. Conserva Telco.
# Para revisar las rutas sin borrar: .\limpiar_datasets_anteriores.ps1 -WhatIf
[CmdletBinding(SupportsShouldProcess=$true, ConfirmImpact='Medium')]
param()
$ErrorActionPreference = 'Stop'
$allowedRoots = @(
 'C:\Users\Marco\Documents\metodos completos',
 'D:\machine learning\lab5',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build'
)
$oldPaths = @(
 'C:\Users\Marco\Documents\metodos completos\1_regresion_lineal\elfec_energia.xlsx',
 'C:\Users\Marco\Documents\metodos completos\2_regresion_polinomial\elfec_energia.xlsx',
 'C:\Users\Marco\Documents\metodos completos\3_regresion_logistica\elfec_energia.xlsx',
 'C:\Users\Marco\Documents\metodos completos\4_knn\elfec_energia.xlsx',
 'C:\Users\Marco\Documents\metodos completos\5_naive_bayes\elfec_energia.xlsx',
 'C:\Users\Marco\Documents\metodos completos\6_arboles_de_decision\elfec_energia.xlsx',
 'C:\Users\Marco\Documents\metodos completos\7_boosting\elfec_energia.xlsx',
 'C:\Users\Marco\Documents\metodos completos\8_adaboost\elfec_energia.xlsx',
 'C:\Users\Marco\Documents\metodos completos\9_gradiant_boosting\elfec_energia.xlsx',
 'C:\Users\Marco\Documents\metodos completos\10_ab_testing\elfec_energia.xlsx',
 'C:\Users\Marco\Documents\metodos completos\11_svm\elfec_energia.xlsx',
 'C:\Users\Marco\Documents\metodos completos\datasets\california_housing_20640_con_texto.xlsx',
 'C:\Users\Marco\Documents\metodos completos\datasets\california_housing_20640.xlsx',
 'C:\Users\Marco\Documents\metodos completos\datasets\elfec_energia_21600.csv',
 'C:\Users\Marco\Documents\metodos completos\datasets\elfec_energia_21600.xlsx',
 'C:\Users\Marco\Documents\metodos completos\datasets\viviendas_sinteticas_10000.xlsx',
 'C:\Users\Marco\Documents\metodos completos\output\pdf\bitacora_elfec.pdf',
 'C:\Users\Marco\Documents\metodos completos\output\pdf\informe_modelos_elfec.pdf',
 'C:\Users\Marco\Documents\metodos completos\sistema\resultados',
 'D:\machine learning\lab5\elfec_app.py',
 'D:\machine learning\lab5\elfec_energia.xlsx',
 'D:\machine learning\lab5\mlartifacts_elfec',
 'D:\machine learning\lab5\resultados_elfec',
 'D:\machine learning\lab5\mlflow_elfec.db',
 'D:\machine learning\lab5\diabetes_model.pkl',
 'D:\machine learning\lab5\mlflow.db',
 'D:\machine learning\lab5\originales_clase',
 'D:\machine learning\lab5\__pycache__\elfec_app.cpython-312.pyc',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\actualizados',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\antes_elfec',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\antes_revision_20260921',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\antes_telco_20260929',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\bolivia_fuentes',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\pdf_elfec',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\pruebas_california',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\pruebas_compactas',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\pruebas_compactas_extra',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\pruebas_elfec',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\pruebas_graficas',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\pruebas_graficas_extra',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\pruebas_sistema',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\revision_20260921',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\actualizar_ficha_elfec.mjs',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\adaptar_elfec.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\adaptar_telco.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\agregar_graficas.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\auditar_elfec.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\auditar_polinomios.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\auditoria_revision.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\build.mjs',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\california_housing_20640.xlsx.inspect.ndjson',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\compactar_graficas.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\copiar_elfec.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\crear_elfec.mjs',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\crear_pdfs_elfec.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\descargar_elfec.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\figuras_informe_elfec.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\graficas_ab.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\graficas_compactas.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\graficas_modelos.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\housing_categorias_fuente.csv',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\inspeccionar_fuentes_bo.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\lab_telco.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\probar_elfec.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\real_antes.png',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\real_categorias_nuevo.png',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\real_categorias.json',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\real_datos_nuevo.png',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\real_Datos.png',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\real_diccionario_nuevo.png',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\real_ficha_nuevo.png',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\real_Ficha.png',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\real.json',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\revisar_y_comentar.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\sintetico_antes.png',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\sintetico_categorias_nuevo.png',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\sintetico_categorias.json',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\sintetico_datos_nuevo.png',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\sintetico_Datos.png',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\sintetico_diccionario_nuevo.png',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\sintetico_ficha_nuevo.png',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\sintetico_Ficha.png',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\sintetico.json',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\test_all_california.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\test_graficas_extra.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\test_sistema_elfec.py',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\update_dictionary.mjs',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\viviendas_sinteticas_10000.xlsx.inspect.ndjson',
 'C:\Users\Marco\.codex\visualizations\2026\09\17\01a0ad59-90cb-70c3-ab3f-6139edbfe974\datasets_build\web_telco.py'
)
# Comprobar todas las rutas antes de eliminar el primer archivo.
$existingPaths = @()
foreach ($oldPath in $oldPaths) {
 if (-not (Test-Path -LiteralPath $oldPath)) { continue }
 $resolved = (Resolve-Path -LiteralPath $oldPath).Path
 $inside = $false
 foreach ($root in $allowedRoots) {
  if ($resolved.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) { $inside = $true }
 }
 if (-not $inside -or $resolved -ne $oldPath) { throw "Ruta no autorizada: $resolved" }
 $entry = Get-Item -LiteralPath $resolved -Force
 if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Enlace inesperado: $resolved" }
 if ($entry.PSIsContainer -and (Get-ChildItem -LiteralPath $resolved -Recurse -Force -Attributes ReparsePoint)) {
  throw "La carpeta contiene enlaces: $resolved"
 }
 $existingPaths += $resolved
}
# Registrar las huellas de las 13 copias de Telco para comprobar que no cambian.
$projectRoot = $allowedRoots[0]
$telcoFiles = @(Join-Path $projectRoot 'telco.csv')
$telcoFiles += @(Get-ChildItem -LiteralPath $projectRoot -Directory | Where-Object Name -Match '^\d+_' | ForEach-Object { Join-Path $_.FullName 'telco.csv' })
$telcoFiles += Join-Path $allowedRoots[1] 'telco.csv'
$telcoHashes = @{}
foreach ($telcoFile in $telcoFiles) { $telcoHashes[$telcoFile] = (Get-FileHash -LiteralPath $telcoFile -Algorithm SHA256).Hash }
$deleted = 0
foreach ($oldPath in $existingPaths) {
 if ($PSCmdlet.ShouldProcess($oldPath, 'Eliminar material de datasets anteriores')) {
  Remove-Item -LiteralPath $oldPath -Recurse -Force
  $deleted++
 }
}
foreach ($telcoFile in $telcoFiles) {
 if ((Get-FileHash -LiteralPath $telcoFile -Algorithm SHA256).Hash -ne $telcoHashes[$telcoFile]) {
  throw "Cambió Telco: $telcoFile"
 }
}
if (-not $WhatIfPreference -and -not @($oldPaths | Where-Object { Test-Path -LiteralPath $_ }).Count) {
 $readme = Join-Path $projectRoot 'sistema\LEEME.md'
 $content = Get-Content -LiteralPath $readme -Raw -Encoding utf8
 $content = $content.Replace(' El historial anterior de ELFEC se conserva en resultados/, separado del nuevo historial.', ' Solo se conserva el historial de Telco.')
 Set-Content -LiteralPath $readme -Value $content -Encoding utf8 -NoNewline
 $readme = Join-Path $allowedRoots[1] 'LEEME.md'
 $content = Get-Content -LiteralPath $readme -Raw -Encoding utf8
 $content = $content.Replace(' Los materiales anteriores de ELFEC se conservan sin usarlos y tienen una copia de respaldo fuera del proyecto. Los originales de diabetes se mantienen en `originales_clase/`.', ' Solo se conservan los datos, modelos y resultados de Telco.')
 Set-Content -LiteralPath $readme -Value $content -Encoding utf8 -NoNewline
}
Write-Output "Eliminados: $deleted archivos o carpetas antiguos. Copias de Telco intactas: $($telcoFiles.Count)."
