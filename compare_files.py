import os

def compare_files_line_by_line(file1_path: str, file2_path: str) -> bool:
    """
    compara dos archivos línea por línea e imprime cualquier diferencia.
    devuelve True si los archivos son diferentes, False en caso contrario.
    """
    are_different = False
    print(f"comparando '{file1_path}' y '{file2_path}'...")

    if not os.path.exists(file1_path):
        print(f"error: archivo '{file1_path}' no encontrado.")
        return True
    if not os.path.exists(file2_path):
        print(f"error: archivo '{file2_path}' no encontrado.")
        return True

    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        for i, (line1, line2) in enumerate(zip(f1, f2)):
            if line1 != line2:
                print(f"se encontró una diferencia en la línea {i + 1}:")
                print(f"  archivo 1: '{line1.strip()}'")
                print(f"  archivo 2: '{line2.strip()}'")
                are_different = True
        
        # verifica si quedan líneas si un archivo es más largo que el otro
        remaining_lines_f1 = f1.readlines()
        remaining_lines_f2 = f2.readlines()

        if remaining_lines_f1:
            print(f"el archivo '{file1_path}' tiene {len(remaining_lines_f1)} líneas más que '{file2_path}'.")
            are_different = True
        if remaining_lines_f2:
            print(f"el archivo '{file2_path}' tiene {len(remaining_lines_f2)} líneas más que '{file1_path}'.")
            are_different = True

    if not are_different:
        print("los archivos son idénticos.")
    else:
        print("los archivos son diferentes.")
        
    return are_different

if __name__ == "__main__":
    # ejemplo de uso
    # crea algunos archivos de prueba
    with open("previous_file.txt", "w") as f:
        f.write("esta es la línea 1\n")
        f.write("esta es la línea 2\n")
        f.write("esta es la línea 3\n")

    with open("current_file_same.txt", "w") as f:
        f.write("esta es la línea 1\n")
        f.write("esta es la línea 2\n")
        f.write("esta es la línea 3\n")

    with open("current_file_different.txt", "w") as f:
        f.write("esta es la línea 1\n")
        f.write("esta es la línea modificada en la línea 2\n")
        f.write("esta es la línea 3\n")
        f.write("esta es una nueva línea en el archivo diferente\n")
    
    with open("current_file_shorter.txt", "w") as f:
        f.write("esta es la línea 1\n")
        f.write("esta es la línea 2\n")


    print("\n--- comparación 1: archivos idénticos ---")
    compare_files_line_by_line("previous_file.txt", "current_file_same.txt")

    print("\n--- comparación 2: archivos diferentes ---")
    compare_files_line_by_line("previous_file.txt", "current_file_different.txt")

    print("\n--- comparación 3: archivo actual más corto ---")
    compare_files_line_by_line("previous_file.txt", "current_file_shorter.txt")

    print("\n--- comparación 4: archivo anterior más corto (mismo que el 3, solo para demostrar) ---")
    compare_files_line_by_line("current_file_shorter.txt", "previous_file.txt")