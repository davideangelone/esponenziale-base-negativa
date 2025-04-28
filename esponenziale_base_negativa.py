import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

# Dati
x_vals = np.linspace(-2, 5, 300)
complex_vals = np.exp(x_vals * np.log(-2 + 0j))  # (-2)^x
real_part = complex_vals.real
imag_part = complex_vals.imag

# Punto da evidenziare
x_evid = 0.5
y_evid = np.exp(x_evid * np.log(-2 + 0j))
re_evid, im_evid = y_evid.real, y_evid.imag
evid_index = np.abs(x_vals - x_evid).argmin()

# Funzione per evidenziare il punto
def evidenzia_punto(ax, x, re_y, im_y, xmin, im_max, re_min):
    offset = 0.5
    z_offset = 0.3

    # Punto nello spazio
    ax.scatter(re_y, im_y, x, color='black', s=60, label=f"x = {x:.2f}", zorder=10)
    ax.text(re_y + offset, im_y + offset, x + z_offset,
            f"x = {x:.2f}\nRe(y) = {re_y:.2f}\nIm(y) = {im_y:.2f}", fontsize=10, color='black', zorder=11,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Proiezione su piano Re(y)-Im(y)
    ax.scatter(re_y, im_y, xmin, color='black', s=40, alpha=0.6, zorder=10)
    ax.text(re_y + offset, im_y + offset, xmin + z_offset,
            f"Re = {re_y:.2f}\nIm = {im_y:.2f}", fontsize=10, color='blue', zorder=11,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Proiezione su piano Re(y)-x
    ax.scatter(re_y, im_max, x, color='black', s=40, alpha=0.6, zorder=10)
    ax.text(re_y + offset, im_max + offset, x + z_offset,
            f"Re = {re_y:.2f}\nx = {x:.2f}", fontsize=10, color='green', zorder=11,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Proiezione su piano Im(y)-x
    ax.scatter(re_min, im_y, x, color='black', s=40, alpha=0.6, zorder=10)
    ax.text(re_min + offset, im_y + offset, x + z_offset,
            f"Im = {im_y:.2f}\nx = {x:.2f}", fontsize=10, color='red', zorder=11,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))


# Setup figura
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Inizializzazione linee
main_line, = ax.plot([], [], [], color='darkorange', linewidth=5, label='y = $(-2)^x$')
proj1_line, = ax.plot([], [], [], color='blue', alpha=0.5, label='Proiezione su piano Re(y)-Im(y)')
proj2_line, = ax.plot([], [], [], color='green', alpha=0.5, label='Proiezione su piano Re(y)-x')
proj3_line, = ax.plot([], [], [], color='red', alpha=0.5, label='Proiezione su piano Im(y)-x')

# Imposta limiti e etichette
ax.set_xlim(real_part.min(), real_part.max())
ax.set_ylim(imag_part.min(), imag_part.max())
ax.set_zlim(x_vals.min(), x_vals.max())
ax.set_xlabel('Re(y) (Parte Reale di y)')
ax.set_ylabel('Im(y) (Parte Immaginaria di y)')
ax.set_zlabel('x (esponente reale)')
ax.set_title(r"Rappresentazione 3D della funzione complessa $y = (-2)^x$, con $x \in \mathbb{R}$", fontsize=16, pad=20)
ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)

# Funzione di inizializzazione
def init():
    for line in [main_line, proj1_line, proj2_line, proj3_line]:
        line.set_data([], [])
        line.set_3d_properties([])
    return main_line, proj1_line, proj2_line, proj3_line

# Funzione per aggiornare ogni frame
def update(frame):
    main_line.set_data(real_part[:frame], imag_part[:frame])
    main_line.set_3d_properties(x_vals[:frame])

    proj1_line.set_data(real_part[:frame], imag_part[:frame])
    proj1_line.set_3d_properties(np.full(frame, x_vals.min()))

    proj2_line.set_data(real_part[:frame], np.full(frame, imag_part.max()))
    proj2_line.set_3d_properties(x_vals[:frame])

    proj3_line.set_data(np.full(frame, real_part.min()), imag_part[:frame])
    proj3_line.set_3d_properties(x_vals[:frame])
    
    if frame == evid_index:
        evidenzia_punto(ax, x_evid, re_evid, im_evid, x_vals.min(), imag_part.max(), real_part.min())

    return main_line, proj1_line, proj2_line, proj3_line

# Creazione e salvataggio del video
ani = animation.FuncAnimation(fig, update, frames=len(x_vals), init_func=init, blit=False, interval=20)
writer = FFMpegWriter(fps=30)
ani.save("esponenziale_base_negativa.mp4", writer=writer)

print("Video salvato come esponenziale_base_negativa.mp4")

# Salva l'ultimo frame come immagine
update(len(x_vals))  # Forza aggiornamento all'ultimo frame
evidenzia_punto(ax, x_evid, re_evid, im_evid, x_vals.min(), imag_part.max(), real_part.min())
plt.savefig("esponenziale_base_negativa.png", dpi=300, bbox_inches='tight')
print("Ultimo frame salvato come 'esponenziale_base_negativa.png'")

# Chiudi la figura dell’animazione
plt.close(fig)

# Crea nuova figura per visualizzazione interattiva
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection='3d')

# Traccia l’intero grafico statico (ultimo frame)
ax2.plot(real_part, imag_part, x_vals, color='darkorange', linewidth=5, label='y = $(-2)^x$')
ax2.plot(real_part, imag_part, np.full_like(x_vals, x_vals.min()), color='blue', alpha=0.5, label='Proiezione su piano Re(y)-Im(y)')
ax2.plot(real_part, np.full_like(x_vals, imag_part.max()), x_vals, color='green', alpha=0.5, label='Proiezione su piano Re(y)-x')
ax2.plot(np.full_like(x_vals, real_part.min()), imag_part, x_vals, color='red', alpha=0.5, label='Proiezione su piano Im(y)-x')

# Etichette e titolo
ax2.set_xlabel('Re(y) (Parte Reale di y)')
ax2.set_ylabel('Im(y) (Parte Immaginaria di y)')
ax2.set_zlabel('x (esponente reale)')
ax2.set_title(r"Rappresentazione 3D della funzione complessa $y = (-2)^x$, con $x \in \mathbb{R}$", fontsize=16, pad=20)
ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)

evidenzia_punto(ax2, x_evid, re_evid, im_evid, x_vals.min(), imag_part.max(), real_part.min())

# Mostra la figura interattiva dell’ultimo frame
plt.show()

