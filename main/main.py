from tvb.simulator.lab import *
import numpy as np
import matplotlib.pyplot as plt


def run_simulation(x0_normal=-2.4, x0_epileptic=-1.6, simulation_length=1000):
    # Set up the Epileptor model
    epileptor_normal = models.Epileptor(x0=np.ones((76)) * x0_normal)
    epileptor_epileptic = models.Epileptor(x0=np.ones((76)) * x0_epileptic)

    # Load the connectivity matrix
    con = connectivity.Connectivity.from_file()

    # Choose a difference coupling function
    coupl = coupling.Difference(a=np.array([1.]))

    # Set up the integrator and monitors
    hiss = noise.Additive(nsig=np.array([0., 0., 0., 0.0003, 0.0003, 0.]))
    heunint = integrators.HeunStochastic(dt=0.05, noise=hiss)

    # Set up the EEG monitor
    rm = region_mapping.RegionMapping.from_file()
    mon_EEG = monitors.EEG.from_file()
    mon_EEG.region_mapping = rm
    period = 1e3 / 256.0  # Set the sampling frequency to 256 Hz
    mon_EEG.period = period

    # Initialize the simulator
    sim_normal = simulator.Simulator(model=epileptor_normal, connectivity=con, coupling=coupl, integrator=heunint,
                                     monitors=[mon_EEG])
    sim_epileptic = simulator.Simulator(model=epileptor_epileptic, connectivity=con, coupling=coupl, integrator=heunint,
                                        monitors=[mon_EEG])
    sim_normal.configure()
    sim_epileptic.configure()

    # Run the simulation
    (_, eeg_normal), = sim_normal.run(simulation_length=simulation_length)
    (_, eeg_epileptic), = sim_epileptic.run(simulation_length=simulation_length)

    # Extract the EEG data
    eeg_normal = eeg_normal[:, 0, :19, 0]  # Extract the first 19 channels
    eeg_epileptic = eeg_epileptic[:, 0, :19, 0]  # Extract the first 19 channels

    return eeg_normal, eeg_epileptic, mon_EEG, period


def plot_data(eeg_normal, eeg_epileptic, mon_EEG, period):
    eeg_sensor_labels = mon_EEG.sensors.labels
    # Plot the EEG data
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(0, 1000, period), eeg_normal)
    plt.xlabel('Time (ms)')
    plt.ylabel('EEG (units)')
    plt.title('Normal EEG data over 1000 ms')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(0, 1000, period), eeg_epileptic)
    plt.xlabel('Time (ms)')
    plt.ylabel('EEG (units)')
    plt.title('Epileptic EEG data over 1000 ms')
    plt.legend()
    plt.show()
    plt.close()


if __name__ == '__main__':
    # Run the simulation and plot the data
    eeg_normal, eeg_epileptic, mon_EEG, period = run_simulation()
    plot_data(eeg_normal, eeg_epileptic, mon_EEG, period)
