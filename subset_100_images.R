

# Función para seleccionar fotos de acuerdo con los pesos
seleccionar_fotos <- function(carpetas, total_fotos = 100) {
  # Contar el número de fotos en cada carpeta
  fotos_por_carpeta <- sapply(carpetas, function(carpeta) length(list.files(carpeta)))
  
  # Calcular los pesos
  total_fotos_disponibles <- sum(fotos_por_carpeta)
  pesos <- fotos_por_carpeta / total_fotos_disponibles
  
  # Seleccionar fotos de acuerdo con los pesos
  fotos_seleccionadas <- c()
  for (i in 1:length(carpetas)) {
    fotos_de_carpeta <- list.files(carpetas[i])
    cantidad_a_seleccionar <- round(pesos[i] * total_fotos)
    
    # Seleccionar aleatoriamente las fotos de esta carpeta
    fotos_seleccionadas <- c(fotos_seleccionadas, sample(fotos_de_carpeta, cantidad_a_seleccionar))
  }
  
  # Si hay menos fotos de las que necesitamos (por redondeo), completar con más fotos aleatorias
  if (length(fotos_seleccionadas) < total_fotos) {
    faltantes <- total_fotos - length(fotos_seleccionadas)
    todas_las_fotos <- unlist(lapply(carpetas, function(carpeta) list.files(carpeta)))
    fotos_seleccionadas <- c(fotos_seleccionadas, sample(todas_las_fotos, faltantes))
  }
  
  return(fotos_seleccionadas)
}

# Función para mover las fotos seleccionadas a una nueva carpeta
mover_fotos <- function(fotos_seleccionadas, carpetas, carpeta_destino) {
  # Crear la carpeta de destino si no existe
  if (!dir.exists(carpeta_destino)) {
    dir.create(carpeta_destino)
  }
  
  # Mover cada foto seleccionada a la nueva carpeta
  for (foto in fotos_seleccionadas) {
    # Buscar la foto en las carpetas de origen
    for (carpeta in carpetas) {
      if (foto %in% list.files(carpeta)) {
        # Mover la foto
        file.copy(file.path(carpeta, foto), file.path(carpeta_destino, foto), overwrite = TRUE)
        break
      }
    }
  }
}

# Directorios de las carpetas
carpetas <- c("~/Downloads/zalando/zalando/sweatshirt", "~/Downloads/zalando/zalando/sweatshirt-female", "~/Downloads/zalando/zalando/hoodies", "~/Downloads/zalando/zalando/hoodies-female", "~/Downloads/zalando/zalando/shirt", "~/Downloads/zalando/zalando/longsleeve")

# Selección de las 100 fotos
fotos_seleccionadas <- seleccionar_fotos(carpetas)

# Ruta de la nueva carpeta donde se guardarán las fotos seleccionadas
carpeta_destino <- "~/Desktop/TFG/subset_100_images"

# Mover las fotos seleccionadas a la nueva carpeta
mover_fotos(fotos_seleccionadas, carpetas, carpeta_destino)



















