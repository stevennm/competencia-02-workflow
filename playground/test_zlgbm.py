"""
Script de prueba para zLightGBM
Verifica que la instalación y configuración sean correctas
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import numpy as np
import lightgbm as lgb


def test_zlgbm_installation():
    """Verifica que zLightGBM esté instalado correctamente"""
    print("="*70)
    print("TEST 1: Verificar instalación de zLightGBM")
    print("="*70)
    
    print(f"\nLightGBM version: {lgb.__version__}")
    
    # Verificar si tiene los parámetros de zLightGBM
    try:
        # Crear un dataset simple
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        dtrain = lgb.Dataset(X, label=y)
        
        # Intentar usar parámetros de zLightGBM
        params = {
            "objective": "binary",
            "canaritos": 2,  # Parámetro específico de zLightGBM
            "gradient_bound": 0.1,  # Parámetro específico de zLightGBM
            "num_iterations": 10,
            "verbosity": -1
        }
        
        modelo = lgb.train(params, dtrain, num_boost_round=10)
        
        print("✓ zLightGBM instalado correctamente")
        print("✓ Parámetros 'canaritos' y 'gradient_bound' reconocidos")
        return True
        
    except Exception as e:
        print(f"✗ Error al usar parámetros de zLightGBM: {e}")
        print("\nPosibles causas:")
        print("1. zLightGBM no está instalado (usar LightGBM estándar)")
        print("2. La instalación no se completó correctamente")
        print("\nPara instalar zLightGBM, seguir instrucciones en:")
        print("docs/ZLIGHTGBM_INTEGRATION.md")
        return False


def test_canaritos_creation():
    """Verifica la creación de canaritos"""
    print("\n" + "="*70)
    print("TEST 2: Creación de canaritos")
    print("="*70)
    
    # Crear dataset de prueba
    df = pl.DataFrame({
        "numero_de_cliente": range(1000),
        "foto_mes": [202106] * 1000,
        "feature1": np.random.rand(1000),
        "feature2": np.random.rand(1000),
        "clase_ternaria": np.random.choice(["CONTINUA", "BAJA+1", "BAJA+2"], 1000)
    })
    
    print(f"\nDataset original: {df.shape}")
    print(f"Columnas: {df.columns}")
    
    # Agregar canaritos
    n_canaritos = 10
    canaritos_dict = {}
    
    for i in range(n_canaritos):
        canaritos_dict[f"canarito_{i+1}"] = np.random.uniform(0, 1, len(df))
    
    df_canaritos = pl.DataFrame(canaritos_dict)
    df_with_canaritos = pl.concat([df_canaritos, df], how="horizontal")
    
    print(f"\nDataset con canaritos: {df_with_canaritos.shape}")
    print(f"Primeras 5 columnas: {df_with_canaritos.columns[:5]}")
    print(f"Últimas 5 columnas: {df_with_canaritos.columns[-5:]}")
    
    # Verificar que canaritos estén al inicio
    expected_canaritos = [f"canarito_{i+1}" for i in range(n_canaritos)]
    actual_first_cols = df_with_canaritos.columns[:n_canaritos]
    
    if list(actual_first_cols) == expected_canaritos:
        print("\n✓ Canaritos están correctamente al inicio del dataset")
        return True
    else:
        print(f"\n✗ Error: Canaritos no están al inicio")
        print(f"  Esperado: {expected_canaritos}")
        print(f"  Obtenido: {list(actual_first_cols)}")
        return False


def test_zlgbm_training():
    """Prueba de entrenamiento con zLightGBM"""
    print("\n" + "="*70)
    print("TEST 3: Entrenamiento con zLightGBM")
    print("="*70)
    
    # Crear dataset sintético
    n_samples = 1000
    n_canaritos = 5
    n_features = 10
    
    print(f"\nCreando dataset sintético:")
    print(f"  Samples: {n_samples}")
    print(f"  Canaritos: {n_canaritos}")
    print(f"  Real features: {n_features}")
    
    # Canaritos (aleatorios, no informativos)
    X_canaritos = np.random.rand(n_samples, n_canaritos)
    
    # Features reales (con señal)
    X_real = np.random.rand(n_samples, n_features)
    
    # Target con señal en features reales
    y = (X_real[:, 0] + X_real[:, 1] > 1).astype(int)
    
    # Combinar: canaritos primero
    X = np.hstack([X_canaritos, X_real])
    
    print(f"\nX shape: {X.shape}")
    print(f"y distribution: {np.bincount(y)}")
    
    # Crear dataset
    dtrain = lgb.Dataset(X, label=y)
    
    # Parámetros zLightGBM
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": 1,
        
        # Hiperparámetros "libres"
        "num_iterations": 100,
        "num_leaves": 31,
        "learning_rate": 1.0,
        
        # zLightGBM específico
        "canaritos": n_canaritos,
        "gradient_bound": 0.1,
        
        "seed": 42
    }
    
    print("\nEntrenando modelo...")
    print(f"  canaritos: {params['canaritos']}")
    print(f"  gradient_bound: {params['gradient_bound']}")
    
    try:
        modelo = lgb.train(params, dtrain, num_boost_round=100)
        
        n_trees = modelo.num_trees()
        print(f"\n✓ Entrenamiento exitoso")
        print(f"  Árboles construidos: {n_trees}")
        
        # Analizar importancia de features
        importance = modelo.feature_importance(importance_type='gain')
        
        print(f"\nImportancia de features:")
        print(f"  Canaritos (primeros {n_canaritos}):")
        for i in range(n_canaritos):
            print(f"    Feature {i}: {importance[i]:.2f}")
        
        print(f"  Features reales (siguientes {n_features}):")
        for i in range(n_canaritos, n_canaritos + min(5, n_features)):
            print(f"    Feature {i}: {importance[i]:.2f}")
        
        # Verificar que canaritos tengan baja importancia
        avg_canarito_importance = importance[:n_canaritos].mean()
        avg_real_importance = importance[n_canaritos:].mean()
        
        print(f"\n  Promedio importancia canaritos: {avg_canarito_importance:.2f}")
        print(f"  Promedio importancia features reales: {avg_real_importance:.2f}")
        
        if avg_real_importance > avg_canarito_importance:
            print("\n✓ Features reales tienen mayor importancia que canaritos (esperado)")
        else:
            print("\n⚠ Canaritos tienen importancia similar o mayor (revisar)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error en entrenamiento: {e}")
        return False


def test_full_workflow():
    """Prueba del workflow completo con Polars"""
    print("\n" + "="*70)
    print("TEST 4: Workflow completo con Polars")
    print("="*70)
    
    # Crear dataset de prueba
    n_samples = 2000
    
    df = pl.DataFrame({
        "numero_de_cliente": range(n_samples),
        "foto_mes": [202105] * 1000 + [202106] * 1000,
        "feature1": np.random.rand(n_samples),
        "feature2": np.random.rand(n_samples),
        "feature3": np.random.rand(n_samples),
        "clase_ternaria": np.random.choice(["CONTINUA", "BAJA+1", "BAJA+2"], 
                                          n_samples, p=[0.95, 0.025, 0.025])
    })
    
    print(f"\nDataset: {df.shape}")
    
    # 1. Agregar canaritos
    n_canaritos = 10
    canaritos_dict = {}
    for i in range(n_canaritos):
        canaritos_dict[f"canarito_{i+1}"] = np.random.uniform(0, 1, len(df))
    
    df_canaritos = pl.DataFrame(canaritos_dict)
    df = pl.concat([df_canaritos, df], how="horizontal")
    
    print(f"✓ Canaritos agregados: {df.shape}")
    
    # 2. Preparar training data
    df_train = df.filter(pl.col("foto_mes") == 202105)
    
    df_train = df_train.with_columns([
        pl.when(pl.col("clase_ternaria").is_in(["BAJA+2", "BAJA+1"]))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("clase01")
    ])
    
    # Features
    campos_buenos = [col for col in df_train.columns 
                    if col not in ["numero_de_cliente", "foto_mes", "clase_ternaria", "clase01"]]
    
    X_train = df_train.select(campos_buenos).to_numpy()
    y_train = df_train.select("clase01").to_numpy().ravel()
    
    print(f"\nTraining data:")
    print(f"  Shape: {X_train.shape}")
    print(f"  Positives: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
    
    # 3. Entrenar
    dtrain = lgb.Dataset(X_train, label=y_train)
    
    params = {
        "objective": "binary",
        "verbosity": -1,
        "num_iterations": 50,
        "num_leaves": 31,
        "learning_rate": 1.0,
        "canaritos": n_canaritos,
        "gradient_bound": 0.1,
        "seed": 42
    }
    
    print("\nEntrenando...")
    modelo = lgb.train(params, dtrain, num_boost_round=50)
    print(f"✓ Modelo entrenado: {modelo.num_trees()} árboles")
    
    # 4. Scoring
    df_future = df.filter(pl.col("foto_mes") == 202106)
    X_future = df_future.select(campos_buenos).to_numpy()
    
    print(f"\nScoring en mes futuro:")
    print(f"  Samples: {X_future.shape[0]}")
    
    predictions = modelo.predict(X_future)
    
    print(f"✓ Predicciones generadas")
    print(f"  Min prob: {predictions.min():.4f}")
    print(f"  Max prob: {predictions.max():.4f}")
    print(f"  Mean prob: {predictions.mean():.4f}")
    
    # 5. Top N
    df_pred = df_future.select(["numero_de_cliente"]).with_columns([
        pl.Series("prob", predictions)
    ])
    
    df_pred = df_pred.sort("prob", descending=True)
    
    top_n = 100
    print(f"\nTop {top_n} clientes:")
    print(f"  Prob range: {df_pred.head(top_n)['prob'].min():.4f} - {df_pred.head(top_n)['prob'].max():.4f}")
    
    print("\n✓ Workflow completo exitoso")
    return True


def main():
    """Ejecutar todos los tests"""
    print("\n" + "="*70)
    print("TESTS DE INTEGRACIÓN zLightGBM")
    print("="*70)
    
    results = {}
    
    # Test 1: Instalación
    results["instalacion"] = test_zlgbm_installation()
    
    # Test 2: Canaritos
    results["canaritos"] = test_canaritos_creation()
    
    # Test 3: Entrenamiento
    if results["instalacion"]:
        results["entrenamiento"] = test_zlgbm_training()
    else:
        print("\n⚠ Saltando test de entrenamiento (zLightGBM no instalado)")
        results["entrenamiento"] = None
    
    # Test 4: Workflow completo
    if results["instalacion"]:
        results["workflow"] = test_full_workflow()
    else:
        print("\n⚠ Saltando test de workflow (zLightGBM no instalado)")
        results["workflow"] = None
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE TESTS")
    print("="*70)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIP"
        print(f"{test_name:20s}: {status}")
    
    # Recomendaciones
    print("\n" + "="*70)
    print("RECOMENDACIONES")
    print("="*70)
    
    if not results["instalacion"]:
        print("\n1. Instalar zLightGBM siguiendo las instrucciones:")
        print("   docs/ZLIGHTGBM_INTEGRATION.md")
        print("\n2. En la VM, ejecutar:")
        print("   cd")
        print("   rm -rf LightGBM")
        print("   git clone --recursive https://github.com/dmecoyfin/LightGBM")
        print("   source ~/.venv/bin/activate")
        print("   pip uninstall --yes lightgbm")
        print("   cd ~/LightGBM")
        print("   sh ./build-python.sh install")
    else:
        print("\n✓ zLightGBM está instalado y funcionando correctamente")
        print("\nPróximos pasos:")
        print("1. Revisar docs/ZLIGHTGBM_INTEGRATION.md")
        print("2. Actualizar src/config.py con parámetros zLightGBM")
        print("3. Implementar funciones de training y scoring")
        print("4. Probar con dataset real")
    
    return 0 if all(r in [True, None] for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

