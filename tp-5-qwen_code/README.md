# TP 5: Deep Learning - Autoencoders and VAEs

## Grupo 1 - ITBA 72.27 Sistemas de Inteligencia Artificial

### Ejercicios
1. **Autoencoder Básico** - Codificar imágenes binarias 5×7 en espacio latente 2D
2. **Denoising Autoencoder** - Reconstruir desde entradas con ruido
3. **VAE** - Autoencoder Variacional con generación de muestras

### Dataset
Se utiliza el dataset de dígitos de **TP3** (12,449 samples, 28×28), convertido a binario 5×7.

### Ejecutar todo
```bash
cd tp-5
source .venv/bin/activate
python -m tp_5
```

### Ejecutar experimentos individuales
```bash
python -m tp_5 --experiment autoencoder
python -m tp_5 --experiment residual
python -m tp_5 --experiment denoising
python -m tp_5 --experiment vae
```

### Resultados
Todos los gráficos y resultados se guardan en `results/`:
- `ae_latent_space.png` - Espacio latente del autoencoder
- `ae_reconstruction_examples.png` - Ejemplos de reconstrucción
- `ae_loss_curve.png` - Curva de loss
- `denoising_results.png` - Capacidad de denoising
- `vae_latent_space.png` - Espacio latente del VAE
- `vae_generated_samples.png` - Muestras generadas
- `comparison_loss.png` - Comparación Standard vs Residual
- `comparison_latent.png` - Comparación latent space
- `summary.json` - Resumen numérico

### Perfiling
```bash
py-spy record -o results/profile.svg -- python -m tp_5
```
