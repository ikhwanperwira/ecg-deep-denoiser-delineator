from matplotlib.font_manager import FontProperties
from pycm import ConfusionMatrix
from matplotlib.lines import Line2D
import warnings
import matplotlib.pyplot as plt
import numpy as np


def plot_delineation(X, y, show=True):
  '''
  X: Vector
  y: Vector
  '''

  y = y.astype(np.uint8)

  # Display parameter
  a = 0.5

  # Get mask of every class
  bl_pred = y == 0
  qrs_pred = y == 1
  t_pred = y == 2
  p_pred = y == 3

  # Plotting
  fig, ax = plt.subplots()
  prev_class = None
  start_idx = 0

  for i in range(len(X)):
    current_class = None
    if bl_pred[i]:
      current_class = 'grey'
    elif p_pred[i]:
      current_class = 'orange'
    elif qrs_pred[i]:
      current_class = 'green'
    elif t_pred[i]:
      current_class = 'purple'

    if current_class != prev_class:
      if prev_class is not None:
        ax.axvspan(start_idx, i, color=prev_class, alpha=a)
      start_idx = i
      prev_class = current_class

  # Fill the last region
  if prev_class is not None:
    ax.axvspan(start_idx, len(X), color=prev_class, alpha=a)

  ax.plot(X, color='blue')
  if show:
    plt.show()

# @title Visualizer Function


def calculate_snr(expected_signal, noised_signal):
  expected_signal = expected_signal.flatten()
  noised_signal = noised_signal.flatten()

  # Calculate the power of the expected signal
  expected_power = np.sum(expected_signal ** 2)

  # Calculate the power of the noise
  noise_power = np.sum((noised_signal - expected_signal) ** 2)
  # Calculate the SNR
  snr = 10 * np.log10(expected_power / noise_power)
  snr_rounded_one_comma = round(snr, 1)

  return snr_rounded_one_comma


# @title Visualizer Function


def calculate_snr(expected_signal, noised_signal):
  expected_signal = expected_signal.flatten()
  noised_signal = noised_signal.flatten()

  # Calculate the power of the expected signal
  expected_power = np.sum(expected_signal ** 2)

  # Calculate the power of the noise
  noise_power = np.sum((noised_signal - expected_signal) ** 2)
  # Calculate the SNR
  snr = 10 * np.log10(expected_power / noise_power)
  snr_rounded_one_comma = round(snr, 1)

  return snr_rounded_one_comma


def plot_delineation_comparison(Xt, yt, Xp, yp, start, stop=None, rec_name='-', lead_name='-', pathology_name='-', fs=360):

  if stop is None:
    stop = -1

  Xt = Xt[start:stop]
  yt = yt[start:stop]
  Xp = Xp[start:stop]
  yp = yp[start:stop]

  # Get mask of every class for prediction
  bl_pred = yp == 0
  qrs_pred = yp == 3
  t_pred = yp == 1
  p_pred = yp == 2

  # Get mask of every class for ground truth
  bl_true = yt == 0
  qrs_true = yt == 3
  t_true = yt == 1
  p_true = yt == 2

  # Create figure with two rows and one column
  fig, (ax1, ax2) = plt.subplots(
      2,
      1,
      figsize=(16, 8),
      sharex=True,
      gridspec_kw={"hspace": 0},
  )

  # Plotting for prediction
  prev_class = None
  start_idx = 0
  for i in range(stop - start):
    current_class = None
    if bl_pred[i]:
      current_class = 'grey'
    elif qrs_pred[i]:
      current_class = 'orange'
    elif t_pred[i]:
      current_class = 'green'
    elif p_pred[i]:
      current_class = 'purple'

    if current_class != prev_class:
      if prev_class is not None:
        ax2.axvspan(start_idx, i, color=prev_class, alpha=0.5)
      start_idx = i
      prev_class = current_class
  # Fill the last region
  if prev_class is not None:
    ax2.axvspan(start_idx, stop - start, color=prev_class, alpha=0.5)

  # Plotting for ground truth
  prev_class = None
  start_idx = 0
  for i in range(stop - start):
    current_class = None
    if bl_true[i]:
      current_class = 'grey'
    elif qrs_true[i]:
      current_class = 'orange'
    elif t_true[i]:
      current_class = 'green'
    elif p_true[i]:
      current_class = 'purple'

    if current_class != prev_class:
      if prev_class is not None:
        ax1.axvspan(start_idx, i, color=prev_class, alpha=0.5)
      start_idx = i
      prev_class = current_class
  # Fill the last region
  if prev_class is not None:
    ax1.axvspan(start_idx, stop - start, color=prev_class, alpha=0.5)

  # First row for ground truth (X_unseen, y_true)
  ax1.plot(Xt, color='blue')
  ax1.set_ylabel('Ground Truth')

  # draw baseline at y=0
  ax1.axhline(y=0, color='red', linestyle='-', lw=0.5)

  # Second row for ground truth (X_pred y_pred)
  ax2.plot(Xp, color='blue')
  ax2.axhline(y=0, color='red', linestyle='-', lw=0.5)
  ax2.set_xlim([0, stop - start])
  ax2.set_ylabel('Prediction')
  ax2.set_xlabel('Index')

  # Retrieve the current x-tick locations
  current_xticks = ax2.get_xticks()

  # Define the new x-tick labels based on absolute start and end
  new_xtick_labels = [int(x + start) for x in current_xticks]

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    ax2.set_xticklabels(new_xtick_labels)

  cm = ConfusionMatrix(actual_vector=yt.flatten(),
                       predict_vector=yp.flatten(), transpose=True)

  # Handle if not number type
  cm.PPV = [0 if not type(x) == float else x for _, x in cm.PPV.items()]
  cm.TPR = [0 if not type(x) == float else x for _, x in cm.TPR.items()]

  # Make length of PPV and TPR consistent, fill with zero if not
  if len(cm.PPV) < 4:
    cm.PPV += [0] * (4 - len(cm.PPV))
  if len(cm.TPR) < 4:
    cm.TPR += [0] * (4 - len(cm.TPR))
  if len(cm.F1) < 4:
    # convert F1 to list and fixed it with length 4
    cm.F1 = list(cm.F1.values())
    cm.F1 += [0] * (4 - len(cm.F1))

  snr_fixed_digit = calculate_snr(Xt, Xp)

  notes_list = [
      f"Recall",
      f"BL         : {cm.TPR[0]:.2f}",
      f"QRS        : {cm.TPR[1]:.2f}",
      f"T          : {cm.TPR[2]:.2f}",
      f"P          : {cm.TPR[3]:.2f}",
      f"",
      f"Rec Name   : {rec_name}",
      f"Lead       : {lead_name}",
      f"Pathology  : {pathology_name}",
      f"Unit       : mV",
      f"Sample Rate: {fs}Hz",
      f"SNR(Pr./GT): {'{:+.1f}'.format(snr_fixed_digit)}dB",
      f"",
      f"Precission",
      f"BL         : {cm.PPV[0]:.2f}",
      f"QRS        : {cm.PPV[1]:.2f}",
      f"T          : {cm.PPV[2]:.2f}",
      f"P          : {cm.PPV[3]:.2f}",
  ]

  # notes_list += catatan

  ax1.set_title(
      f"F1-Score | BL: {cm.F1[0]:.2f} | QRS: {cm.F1[1]:.2f} | T: {cm.F1[2]:.2f} | P: {cm.F1[3]:.2f}")

  code_font = FontProperties(
      family='monospace', style='normal', variant='normal', size=8)

  for i, note in enumerate(notes_list[:]):
    plt.text(1.01, 0.95 - i * 0.1, note, transform=ax1.transAxes,
             fontsize=10, va='top', ha='left', fontproperties=code_font)

  plt.subplots_adjust(top=0.5)

  # Create custom Line2D objects with desired colors
  custom_lines = [
      Line2D([0], [0], color='grey', lw=4, alpha=0.5),
      Line2D([0], [0], color='orange', lw=4, alpha=0.5),
      Line2D([0], [0], color='green', lw=4, alpha=0.5),
      Line2D([0], [0], color='purple', lw=4, alpha=0.5)
  ]

  # Add legend with custom lines
  ax1.legend(custom_lines, ['BL', 'QRS', 'T', 'P'], loc='upper left')

  # get min or max value of amplitude of Xp and Xt
  min_value = min(min(Xp), min(Xt))
  max_value = max(max(Xp), max(Xt))

  ax1.set_ylim([min_value, max_value])
  ax2.set_ylim([min_value, max_value])

  # y label use fixed character, prepend + if positive, - if negative, and always display float with one comma
  ax1.get_yaxis().set_major_formatter(
      plt.FuncFormatter(lambda y, _: '{:+.1f}'.format(y)))
  ax2.get_yaxis().set_major_formatter(
      plt.FuncFormatter(lambda y, _: '{:+.1f}'.format(y)))

  # twinned version of ax1
  ax1_twin = ax1.twiny()  # for display x-axis at top

  # Display for second
  ax1_xticks = np.unique(np.linspace(start, stop, num=5, dtype=int))
  ax1_twin.set_xlabel('Second')
  ax1_twin.set_xlim(ax1.get_xlim())
  ax1_twin.set_xticks(ax1_xticks)
  ax1_twin.set_xticklabels(np.round(np.linspace(
      start/fs, stop/fs, num=len(ax1_xticks)), 3))

  # ax2 set xtick based on second converted to index rounded
  ax2.set_xticks(ax1_xticks)
  ax2.set_xticklabels(np.round(np.linspace(
      start, stop, num=len(ax1_xticks)), 3).astype(int))

  # bold for y label text
  ax1.get_yaxis().label.set_fontweight('bold')
  ax2.get_yaxis().label.set_fontweight('bold')
  ax1_twin.get_xaxis().label.set_fontweight('bold')
  ax2.get_xaxis().label.set_fontweight('bold')

  # bold all text
  for text in ax1.get_xticklabels() + ax1.get_yticklabels():
    text.set_fontweight('bold')

  for text in ax2.get_xticklabels() + ax2.get_yticklabels():
    text.set_fontweight('bold')

  for text in ax1_twin.get_xticklabels() + ax1_twin.get_yticklabels():
    text.set_fontweight('bold')

  # bold title
  ax1.title.set_fontweight('bold')

  # show grid
  ax1.grid(True, which='both', linestyle='-',
           linewidth=0.5, color='black', alpha=0.2)
  ax2.grid(True, which='both', linestyle='-',
           linewidth=0.5, color='black', alpha=0.2)

  plt.show()
