#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DETECTOR DE CUELLOS DE BOTELLA EN PROCESOS INDUSTRIALES
Optimización de procesos mediante autovalores y autovectores

Curso: Álgebra para Ingeniería
Proyecto Final - Universidad Científica del Sur

VERSIÓN CORREGIDA:
- Corregido error de lógica en identificación del cuello de botella
- El cuello de botella se identifica SOLO en etapas productivas (A, H, E, R)
- NO en estados absorbentes (V, D)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import seaborn as sns
from datetime import datetime
import json

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DetectorCuelloBotellaCorregido:
    """
    Clase que implementa la metodología del informe para detectar cuellos de botella
    usando autovalores y autovectores calculados con NumPy/SciPy
    
    CORRECCIÓN: El cuello de botella se identifica SOLO en etapas productivas
    """
    
    def __init__(self):
        self.etapas = ['Amasado', 'Horneado', 'Empaque', 'Reproceso', 'Venta', 'Descarte']
        self.etapas_abrev = ['A', 'H', 'E', 'R', 'V', 'D']
        self.etapas_productivas = ['Amasado', 'Horneado', 'Empaque', 'Reproceso']  # Solo estas pueden ser cuellos de botella
        self.matriz_transicion = None
        self.autovalores = None
        self.autovectores = None
        self.autovalor_dominante = None
        self.autovector_dominante = None
        self.cuello_botella_idx = None
        self.cuello_botella_nombre = None
        self.datos_flujo = {}
        
    def ingresar_datos_flujo(self):
        """
        Permite al usuario ingresar los datos de flujo del proceso
        """
        print("\n" + "="*60)
        print("INGRESO DE DATOS DE FLUJO DE PRODUCCIÓN")
        print("="*60)
        print("Ingrese la cantidad de productos que fluyen entre cada etapa:")
        print("(Presione Enter para usar valores de ejemplo de la panadería)\n")
        
        # Valores por defecto (ejemplo de la panadería del informe)
        valores_default = {
            'A_to_H': 1000,
            'H_to_E': 944,
            'H_to_R': 56,
            'E_to_V': 921,
            'E_to_R': 23,
            'R_to_H': 59,
            'R_to_D': 20
        }
        
        self.datos_flujo = {}
        
        try:
            # Amasado → Horneado
            val = input(f"Productos de Amasado a Horneado [{valores_default['A_to_H']}]: ")
            self.datos_flujo['A_to_H'] = int(val) if val.strip() else valores_default['A_to_H']
            
            # Horneado → Empaque
            val = input(f"Productos de Horneado a Empaque [{valores_default['H_to_E']}]: ")
            self.datos_flujo['H_to_E'] = int(val) if val.strip() else valores_default['H_to_E']
            
            # Horneado → Reproceso
            val = input(f"Productos de Horneado a Reproceso [{valores_default['H_to_R']}]: ")
            self.datos_flujo['H_to_R'] = int(val) if val.strip() else valores_default['H_to_R']
            
            # Empaque → Venta
            val = input(f"Productos de Empaque a Venta [{valores_default['E_to_V']}]: ")
            self.datos_flujo['E_to_V'] = int(val) if val.strip() else valores_default['E_to_V']
            
            # Empaque → Reproceso
            val = input(f"Productos de Empaque a Reproceso [{valores_default['E_to_R']}]: ")
            self.datos_flujo['E_to_R'] = int(val) if val.strip() else valores_default['E_to_R']
            
            # Reproceso → Horneado
            val = input(f"Productos de Reproceso a Horneado [{valores_default['R_to_H']}]: ")
            self.datos_flujo['R_to_H'] = int(val) if val.strip() else valores_default['R_to_H']
            
            # Reproceso → Descarte
            val = input(f"Productos de Reproceso a Descarte [{valores_default['R_to_D']}]: ")
            self.datos_flujo['R_to_D'] = int(val) if val.strip() else valores_default['R_to_D']
            
        except ValueError:
            print("Error: Por favor ingrese valores numéricos válidos.")
            self.datos_flujo = valores_default.copy()
        
        print(f"\n✓ Datos ingresados correctamente")
        self._mostrar_resumen_flujo()
    
    def _mostrar_resumen_flujo(self):
        """Muestra un resumen de los datos de flujo ingresados"""
        print("\n📊 RESUMEN DEL FLUJO DE PRODUCCIÓN:")
        print("-" * 40)
        total_entrada = self.datos_flujo['A_to_H']
        total_salida = self.datos_flujo['E_to_V']
        total_reproceso = self.datos_flujo['H_to_R'] + self.datos_flujo['E_to_R']
        total_descarte = self.datos_flujo['R_to_D']
        
        print(f"Productos iniciales:     {total_entrada:,}")
        print(f"Productos vendidos:      {total_salida:,}")
        print(f"Productos en reproceso:  {total_reproceso:,}")
        print(f"Productos descartados:   {total_descarte:,}")
        print(f"Eficiencia del proceso:  {(total_salida/total_entrada*100):.1f}%")
    
    def construir_matriz_transicion(self):
        """
        Construye la matriz de transición estocástica a partir de los datos de flujo
        """
        print("\n" + "="*60)
        print("CONSTRUCCIÓN DE MATRIZ DE TRANSICIÓN")
        print("="*60)
        
        # Inicializar matriz 6x6 con ceros
        self.matriz_transicion = np.zeros((6, 6), dtype=float)
        
        # Flujo de Amasado (A) → Horneado (H) - 100%
        self.matriz_transicion[0, 1] = 1.0
        
        # Flujo de Horneado (H)
        total_h = self.datos_flujo['H_to_E'] + self.datos_flujo['H_to_R']
        if total_h > 0:
            self.matriz_transicion[1, 2] = self.datos_flujo['H_to_E'] / total_h  # A empaque
            self.matriz_transicion[1, 3] = self.datos_flujo['H_to_R'] / total_h  # A reproceso
        
        # Flujo de Empaque (E)
        total_e = self.datos_flujo['E_to_V'] + self.datos_flujo['E_to_R']
        if total_e > 0:
            self.matriz_transicion[2, 4] = self.datos_flujo['E_to_V'] / total_e  # A venta
            self.matriz_transicion[2, 3] = self.datos_flujo['E_to_R'] / total_e  # A reproceso
        
        # Flujo de Reproceso (R)
        total_r = self.datos_flujo['R_to_H'] + self.datos_flujo['R_to_D']
        if total_r > 0:
            self.matriz_transicion[3, 1] = self.datos_flujo['R_to_H'] / total_r  # A horneado
            self.matriz_transicion[3, 5] = self.datos_flujo['R_to_D'] / total_r  # A descarte
        
        # Estados absorbentes (según el informe)
        self.matriz_transicion[4, 4] = 1.0  # Venta se queda en venta
        self.matriz_transicion[5, 5] = 1.0  # Descarte se queda en descarte
        
        self._mostrar_matriz()
    
    def _mostrar_matriz(self):
        """Muestra la matriz de transición de forma formateada"""
        print("\n📐 MATRIZ DE TRANSICIÓN (6×6):")
        print("-" * 50)
        
        # Encabezado
        print("      ", end="")
        for etapa in self.etapas_abrev:
            print(f"{etapa:8}", end="")
        print()
        
        # Filas de la matriz
        for i, fila in enumerate(self.matriz_transicion):
            print(f"{self.etapas_abrev[i]:4}", end="")
            for valor in fila:
                print(f"{valor:8.3f}", end="")
            print(f"  {self.etapas[i]}")
        
        # Verificar que es estocástica
        sumas_filas = np.sum(self.matriz_transicion, axis=1)
        print(f"\n✓ Verificación (suma de filas = 1.0): {np.allclose(sumas_filas, 1.0)}")
    
    def calcular_autovalores_scipy(self):
        """
        Calcula autovalores y autovectores usando SciPy (según el informe)
        """
        print("\n" + "="*60)
        print("CÁLCULO DE AUTOVALORES Y AUTOVECTORES")
        print("Usando SciPy (según metodología del informe)")
        print("="*60)
        
        # Calcular autovalores y autovectores usando SciPy
        self.autovalores, self.autovectores = linalg.eig(self.matriz_transicion)
        
        # Convertir a reales (eliminando parte imaginaria despreciable)
        self.autovalores = np.real(self.autovalores)
        self.autovectores = np.real(self.autovectores)
        
        # Ordenar por magnitud (mayor a menor) - importante para encontrar el dominante
        idx = np.argsort(np.abs(self.autovalores))[::-1]
        self.autovalores = self.autovalores[idx]
        self.autovectores = self.autovectores[:, idx]
        
        print("\\n📈 AUTOVALORES ENCONTRADOS:")
        print("-" * 40)
        for i, av in enumerate(self.autovalores):
            print(f"   λ{i+1} = {av:.6f}")
        
        # Identificar el autovalor dominante (el de mayor magnitud)
        self.autovalor_dominante = self.autovalores[0]
        self.autovector_dominante = self.autovectores[:, 0]
        
        # Normalizar el autovector dominante para que sume 1
        # Esto es crucial para la interpretación correcta
        self.autovector_dominante = np.abs(self.autovector_dominante)
        self.autovector_dominante = self.autovector_dominante / self.autovector_dominante.sum()
        
        print(f"\\n✓ Autovalor dominante: λ_max = {self.autovalor_dominante:.6f}")
        print(f"✓ Autovector dominante normalizado:")
        for i, valor in enumerate(self.autovector_dominante):
            print(f"   {self.etapas[i]:12}: {valor:.6f}")
    
    def identificar_cuello_botella(self):
        """
        Identifica el cuello de botella basado en el autovector dominante
        **CORREGIDO**: Solo considera etapas productivas, no estados absorbentes
        """
        print("\\n" + "="*60)
        print("IDENTIFICACIÓN DEL CUELLO DE BOTELLA")
        print("Basado en el autovector dominante")
        print("(CORREGIDO - Solo etapas productivas)")
        print("="*60)
        
        # **CORRECCIÓN IMPORTANTE**
        # Solo consideramos las etapas productivas: Amasado, Horneado, Empaque, Reproceso
        # NO consideramos Venta y Descarte porque son estados absorbentes
        # El cuello de botella está en el proceso, no en los resultados finales
        
        etapas_productivas = self.etapas[:4]  # Primeras 4 etapas
        valores_productivos = self.autovector_dominante[:4]  # Sus valores en el autovector
        
        # Encontrar el valor máximo entre las etapas productivas
        # Este es el cuello de botella según la metodología
        max_idx_productivo = np.argmax(valores_productivos)
        self.cuello_botella_idx = max_idx_productivo  # Índice en las etapas productivas
        self.cuello_botella_nombre = etapas_productivas[max_idx_productivo]
        
        print("📊 DISTRIBUCIÓN ESTACIONARIA DEL FLUJO:")
        print("(Basada en el autovector dominante normalizado)")
        print("-" * 50)
        
        for i, (etapa, valor) in enumerate(zip(self.etapas, self.autovector_dominante)):
            porcentaje = valor * 100
            barra = "█" * int(porcentaje * 2)
            
            if i < 4 and i == self.cuello_botella_idx:  # Solo marcar si es etapa productiva
                print(f"🔴 {etapa:12}: {porcentaje:6.2f}% {barra} ← CUELLO DE BOTELLA")
            elif i < 4:  # Etapa productiva normal
                print(f"   {etapa:12}: {porcentaje:6.2f}% {barra}")
            else:  # Estados absorbentes
                print(f"📤 {etapa:12}: {porcentaje:6.2f}% {barra} ← Estado absorbente")
        
        print(f"\\n🎯 Cuello de botella identificado: {self.cuello_botella_nombre}")
        print(f"   Este proceso concentra {(valores_productivos[max_idx_productivo] * 100):.2f}% del flujo en equilibrio")
        print(f"   Requiere optimización prioritaria")
        print(f"   (Los estados absorbentes Venta y Descarte NO pueden ser cuellos de botella)")
    
    def analizar_estabilidad_sistema(self):
        """
        Analiza la estabilidad del sistema basado en el autovalor dominante
        """
        print("\\n" + "="*60)
        print("ANÁLISIS DE ESTABILIDAD DEL SISTEMA")
        print("="*60)
        
        if abs(self.autovalor_dominante - 1.0) < 0.001:
            print("✅ SISTEMA ESTABLE")
            print("   El flujo de producción se mantiene sin pérdidas significativas")
            print("   λ ≈ 1.0 indica un sistema en equilibrio")
            estabilidad = "ESTABLE"
        elif self.autovalor_dominante < 1.0:
            perdida = (1 - self.autovalor_dominante) * 100
            print("⚠️  SISTEMA CON PÉRDIDAS")
            print(f"   Se pierde {perdida:.2f}% del flujo en cada ciclo")
            print("   λ < 1.0 indica pérdidas en el sistema")
            estabilidad = "CON_PÉRDIDAS"
        else:
            print("⚠️  SISTEMA INESTABLE")
            print("   λ > 1.0 puede indicar crecimiento descontrolado o errores en datos")
            estabilidad = "INESTABLE"
        
        return estabilidad
    
    def generar_recomendaciones(self):
        """
        Genera recomendaciones específicas basadas en el análisis
        """
        print("\\n" + "="*60)
        print("RECOMENDACIONES DE OPTIMIZACIÓN")
        print("="*60)
        
        recomendaciones = []
        
        # Recomendación principal basada en el cuello de botella
        recomendaciones.append(f"🎯 PRIORIDAD ALTA: Optimizar el proceso de {self.cuello_botella_nombre}")
        recomendaciones.append(f"   - Aumentar la capacidad de procesamiento en {self.cuello_botella_nombre}")
        recomendaciones.append(f"   - Implementar mejores controles de calidad en esta etapa")
        recomendaciones.append(f"   - Capacitar al personal específicamente para {self.cuello_botella_nombre}")
        
        # Análisis de eficiencia general
        total_inicial = self.datos_flujo['A_to_H']
        total_final = self.datos_flujo['E_to_V']
        eficiencia = (total_final / total_inicial) * 100
        
        if eficiencia < 90:
            recomendaciones.append(f"📈 Mejorar eficiencia general: {eficiencia:.1f}%")
            recomendaciones.append("   - Revisar todos los procesos para identificar pérdidas")
        
        # Análisis de reproceso
        total_reproceso = self.datos_flujo['H_to_R'] + self.datos_flujo['E_to_R']
        tasa_reproceso = (total_reproceso / total_inicial) * 100
        
        if tasa_reproceso > 5:
            recomendaciones.append(f"🔄 Reducir tasa de reproceso: {tasa_reproceso:.1f}%")
            recomendaciones.append("   - Implementar inspección más rigurosa en etapas anteriores")
        
        # Recomendaciones generales
        recomendaciones.append("📊 Implementar monitoreo continuo de KPIs")
        recomendaciones.append("🔄 Realizar análisis periódicos usando esta metodología")
        
        # Mostrar recomendaciones
        for i, rec in enumerate(recomendaciones, 1):
            print(f"{i:2d}. {rec}")
        
        return recomendaciones
    
    def crear_visualizaciones(self):
        """
        Crea visualizaciones gráficas del análisis según la metodología del informe
        """
        print("\\n" + "="*60)
        print("GENERANDO VISUALIZACIONES")
        print("="*60)
        
        # Crear figura con subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis de Cuello de Botella - Panadería Artesanal\\n(Usando Autovalores y Autovectores - Metodología del Informe)', 
                     fontsize=16, fontweight='bold')
        
        # 1. Distribución del flujo basada en el autovector dominante
        # **CORREGIDO**: Solo destacar el cuello de botella en etapas productivas
        colores = []
        for i in range(len(self.etapas)):
            if i < 4 and i == self.cuello_botella_idx:  # Si es etapa productiva y es el cuello de botella
                colores.append('red')
            elif i < 4:  # Etapa productiva normal
                colores.append('skyblue')
            else:  # Estados absorbentes
                colores.append('lightgray')
        
        barras = ax1.bar(self.etapas, self.autovector_dominante * 100, color=colores)
        ax1.set_title('Distribución Estacionaria del Flujo\\n(Autovector Dominante)')
        ax1.set_ylabel('Porcentaje del Flujo (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Añadir valores en las barras
        for i, (barra, valor) in enumerate(zip(barras, self.autovector_dominante)):
            altura = valor * 100
            ax1.text(barra.get_x() + barra.get_width()/2, altura + 1, 
                    f'{altura:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Destacar cuello de botella (solo si es etapa productiva)
        if self.cuello_botella_idx < 4:
            ax1.annotate(f'Cuello de Botella\\n{self.cuello_botella_nombre}\\n{(self.autovector_dominante[self.cuello_botella_idx] * 100):.1f}%', 
                        xy=(self.cuello_botella_idx, 
                            self.autovector_dominante[self.cuello_botella_idx] * 100),
                        xytext=(self.cuello_botella_idx + 1, 
                               self.autovector_dominante[self.cuello_botella_idx] * 100 + 10),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=10, ha='center', color='red', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 2. Matriz de transición como heatmap
        im = ax2.imshow(self.matriz_transicion, cmap='Blues', aspect='auto')
        ax2.set_title('Matriz de Transición Estocástica')
        ax2.set_xticks(range(len(self.etapas_abrev)))
        ax2.set_yticks(range(len(self.etapas_abrev)))
        ax2.set_xticklabels(self.etapas_abrev)
        ax2.set_yticklabels(self.etapas_abrev)
        
        # Añadir valores a la matriz
        for i in range(len(self.etapas)):
            for j in range(len(self.etapas)):
                valor = self.matriz_transicion[i, j]
                if valor > 0.001:
                    ax2.text(j, i, f'{valor:.3f}', 
                            ha='center', va='center', fontsize=8, fontweight='bold')
        
        plt.colorbar(im, ax=ax2, label='Probabilidad')
        
        # 3. Flujo de producción (cantidades reales)
        flujo_data = {
            'A→H': self.datos_flujo['A_to_H'],
            'H→E': self.datos_flujo['H_to_E'],
            'H→R': self.datos_flujo['H_to_R'],
            'E→V': self.datos_flujo['E_to_V'],
            'E→R': self.datos_flujo['E_to_R'],
            'R→H': self.datos_flujo['R_to_H'],
            'R→D': self.datos_flujo['R_to_D']
        }
        
        colores_flujo = ['green', 'blue', 'orange', 'green', 'orange', 'blue', 'red']
        barras_flujo = ax3.bar(range(len(flujo_data)), list(flujo_data.values()), color=colores_flujo)
        ax3.set_title('Flujo de Productos entre Etapas')
        ax3.set_ylabel('Cantidad de Productos')
        ax3.set_xticks(range(len(flujo_data)))
        ax3.set_xticklabels(list(flujo_data.keys()), rotation=45, ha='right')
        
        # Añadir valores en las barras
        for barra, valor in zip(barras_flujo, flujo_data.values()):
            ax3.text(barra.get_x() + barra.get_width()/2, valor + 10, 
                    f'{valor}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Análisis de eficiencia por etapa
        eficiencias = []
        nombres_eficiencia = []
        
        # Eficiencia por etapa
        if self.datos_flujo['A_to_H'] > 0:
            eficiencia_horneado = (self.datos_flujo['H_to_E'] / self.datos_flujo['A_to_H']) * 100
            eficiencias.append(eficiencia_horneado)
            nombres_eficiencia.append('Horneado')
        
        if self.datos_flujo['H_to_E'] > 0:
            eficiencia_empaque = (self.datos_flujo['E_to_V'] / self.datos_flujo['H_to_E']) * 100
            eficiencias.append(eficiencia_empaque)
            nombres_eficiencia.append('Empaque')
        
        if self.datos_flujo['H_to_R'] + self.datos_flujo['E_to_R'] > 0:
            eficiencia_reproceso = (self.datos_flujo['R_to_H'] / (self.datos_flujo['H_to_R'] + self.datos_flujo['E_to_R'])) * 100
            eficiencias.append(eficiencia_reproceso)
            nombres_eficiencia.append('Reproceso')
        
        colores_ef = ['green' if ef > 90 else 'orange' if ef > 70 else 'red' for ef in eficiencias]
        
        barras_ef = ax4.bar(nombres_eficiencia, eficiencias, color=colores_ef)
        ax4.set_title('Eficiencia por Etapa')
        ax4.set_ylabel('Eficiencia (%)')
        ax4.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='Meta: 90%')
        ax4.legend()
        
        # Añadir valores en las barras
        for barra, ef in zip(barras_ef, eficiencias):
            ax4.text(barra.get_x() + barra.get_width()/2, ef + 2, f'{ef:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Guardar gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'analisis_cuello_botella_CORREGIDO_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado como: {filename}")
        
        plt.show()
        
        return fig
    
    def generar_reporte(self):
        """
        Genera un reporte completo del análisis según la metodología del informe
        """
        print("\\n" + "="*80)
        print("📋 REPORTE FINAL - DETECCIÓN DE CUELLOS DE BOTELLA")
        print("(VERSIÓN CORREGIDA - Metodología del Informe)")
        print("="*80)
        
        reporte = {
            'fecha_analisis': datetime.now().isoformat(),
            'version': 'CORREGIDA - Identificación correcta del cuello de botella',
            'metodologia': 'Autovalores y autovectores usando SciPy/NumPy',
            'datos_flujo': self.datos_flujo,
            'cuello_botella': self.cuello_botella_nombre,
            'cuello_botella_porcentaje': float(self.autovector_dominante[self.cuello_botella_idx] * 100),
            'autovalor_dominante': float(self.autovalor_dominante),
            'distribucion_flujo': {
                etapa: float(porcentaje) 
                for etapa, porcentaje in zip(self.etapas, self.autovector_dominante)
            },
            'eficiencia_general': float((self.datos_flujo['E_to_V'] / self.datos_flujo['A_to_H']) * 100),
            'tasa_reproceso': float(((self.datos_flujo['H_to_R'] + self.datos_flujo['E_to_R']) / self.datos_flujo['A_to_H']) * 100),
            'estabilidad_sistema': 'ESTABLE' if abs(self.autovalor_dominante - 1.0) < 0.001 else 'INESTABLE',
            'recomendaciones': [
                f"Optimizar el proceso de {self.cuello_botella_nombre}",
                "Implementar controles de calidad más estrictos",
                "Capacitar al personal en las etapas críticas",
                "Monitorear continuamente los KPIs de producción"
            ]
        }
        
        # Guardar reporte como JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'reporte_cuello_botella_CORREGIDO_{timestamp}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Reporte guardado como: {filename}")
        
        # Mostrar resumen ejecutivo
        print(f"\\n📊 RESUMEN EJECUTIVO:")
        print(f"   • Versión: {reporte['version']}")
        print(f"   • Metodología: {reporte['metodologia']}")
        print(f"   • Cuello de botella: {self.cuello_botella_nombre}")
        print(f"   • Porcentaje de flujo: {reporte['cuello_botella_porcentaje']:.2f}%")
        print(f"   • Eficiencia general: {reporte['eficiencia_general']:.1f}%")
        print(f"   • Autovalor dominante: {self.autovalor_dominante:.6f}")
        print(f"   • Sistema: {reporte['estabilidad_sistema']}")
        
        return reporte

def main():
    """
    Función principal del programa usando la metodología del informe
    """
    print("🍞 DETECTOR DE CUELLOS DE BOTELLA EN PANADERÍA ARTESANAL")
    print("Optimización de procesos mediante Álgebra Lineal")
    print("VERSIÓN CORREGIDA - Identificación correcta del cuello de botella")
    print("="*60)
    print("Este programa utiliza autovalores y autovectores calculados")
    print("con SciPy para identificar los cuellos de botella en procesos")
    print("de producción, CORRIGIENDO el error de identificación.")
    print("\\n⚠️  IMPORTANTE: El cuello de botella se identifica SOLO en las")
    print("etapas productivas (Amasado, Horneado, Empaque, Reproceso)")
    print("NO en los estados absorbentes (Venta, Descarte).")
    
    # Crear instancia del detector
    detector = DetectorCuelloBotellaCorregido()
    
    try:
        # Ejecutar análisis completo según la metodología del informe
        detector.ingresar_datos_flujo()
        detector.construir_matriz_transicion()
        detector.calcular_autovalores_scipy()
        detector.analizar_estabilidad_sistema()
        detector.identificar_cuello_botella()
        detector.generar_recomendaciones()
        detector.crear_visualizaciones()
        detector.generar_reporte()
        
        print("\\n" + "="*60)
        print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("(VERSIÓN CORREGIDA)")
        print("="*60)
        
    except Exception as e:
        print(f"\\n❌ Error durante el análisis: {str(e)}")
        print("Por favor, verifique los datos ingresados e intente nuevamente.")

if __name__ == "__main__":
    main()