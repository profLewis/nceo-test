# -*- coding: utf-8 -*-

"""
    Program to 
    ============================================================
    
    Language: Python
    
    Authors
    -------
    Bethan Harris
    Assimila
    c/o University of Reading
    bethan.harris@reading.ac.uk
       
    BASED ON CODE WRITTEN BY PETER JAN VAN LEEUWEN
    (www.nceo.ac.uk/PFtools/code/PFModel.py)     
    
    With thanks to Melanie Ades
       
    ============================================================
    
    Use the email address above for correspondance relating to this code
      
"""
########################################################################
##   IMPORT LIBRARIES
########################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

#----------------------------------------------------------------------#
#-- NON-LINEAR MODEL --------------------------------------------------#
#----------------------------------------------------------------------#
def NonLinModel(old, k, force=0):
    return 0.5*old + 25*old/(1+old**2) + 8*np.cos(0.8*k) + force

#----------------------------------------------------------------------#
#-- RUNNING THE CODE --------------------------------------------------#
#----------------------------------------------------------------------#
# Number of timesteps is
Timesteps = 100


# The number of particles we will use is
N = 10

    
#The TRUTH ####################################################################
#The initial temperature is:
Truth = np.zeros(Timesteps)
Truth[0] = -8.0 # degrees celcius at start of run

#truth run
for t in range(1,Timesteps):
    Truth[t] = Truth[t]=NonLinModel(Truth[t-1],t,np.random.randn())



##The Observations ############################################################
Observations = np.zeros(Timesteps)*np.NaN

#Number of observations:
obs_no = 5.0

## These will be scattered around the truth with a variance of:
obs_var = 0.5 

#timesteps interval between observations
obs_interval = np.round(Timesteps/(obs_no+1))

#timesteps at which observations take place
obs_timesteps = np.linspace(obs_interval,obs_interval*obs_no,obs_no)
obs_timesteps = np.round(obs_timesteps,0)


#generate observations from Truth
for t in obs_timesteps:
    Observations[t] = Truth[t] + np.sqrt(obs_var) * np.random.randn()


## The Particles ##############################################################
#model error variance
model_var = 0.05

#strength of pull
# this should be no bigger than (obs_var / model_var) to avoid over-pulling and instability
pull_strength = 5


# fraction of particles to be reatined at each observation timestep
particles_retained = 0.8

# The initial temperatures of each particle are evenly spread between 
P = np.zeros([N])
Particles = np.zeros([N,Timesteps])
Particles[:,0] = np.linspace(-20,20,N)

#The weights
W = np.zeros([N])
Weights = np.zeros([N,Timesteps])
Weights[:,0] = 1.0/N

Log_Weights = np.zeros([N,Timesteps])


W_Proposal = np.zeros([N])
W_equalising = np.zeros([N,Timesteps]) # empty array for proposal weights

#statistics for forecasting
Particles_mean = np.zeros(Timesteps)

#For plotting only
Colour_index = np.zeros([N,Timesteps])
Colour_index[:,0] = np.array(range(N))/float(N)
Particles_before_resampling = np.zeros([N,Timesteps])*np.NaN #used to record particles before resampling

#now loop each particle over our timesteps
for t in range(1,Timesteps):

    for n in range(N):
        
        #set up the random error for the Particles:
        force = np.random.randn()*np.sqrt(model_var) 
        
        # calculate the particles' positions################
        Particles[n,t]=NonLinModel(Particles[n,t-1],t,force)   
        
        #How many previous observations have we had?
        previous_obs_no = np.sum(np.isfinite(Observations[:t]))
        
        #Set up the pull term:
        # set the pull fraction initially to be zero
        pull_fraction = 0.0
        # define the pull fraction to be equal to how far away in time we are 
        # from the next observation. 0 is the furthest (previous timestep) 1 is 
        # the closest (current timestep)
        pull_fraction = (1.0*t - obs_interval*previous_obs_no)/(1.0*obs_interval)
        # Set up the reduced pull fraction, this takes the fraction 
        # calculated above and reduced the value by 0.3 The pull strength
        # starts at 0 and can peak at a maximum of 0.7 before the observation timestep.
        if pull_fraction > 0.3:
            pull_fraction_reduced = pull_fraction-0.3
        else:
            pull_fraction_reduced = 0.0
            
        # Now we calculate the pull term  to be included into the model
        # equations using the Particles, the observations, the reduced
        # pull fraction, the model and observation errors and a user-defined 
        # pull strength.
            
        #first make sure that we have an observation to pull towards 
        if (previous_obs_no+1)*obs_interval <= Timesteps:
            pull = pull_strength * model_var / obs_var * pull_fraction_reduced * (Observations[(previous_obs_no+1)*obs_interval] - Particles[n,t])
            
            # Add this pull term on to the Particle value 
            Particles[n,t] += pull            
            
            # Set up the log of the portion of the weights which come from
            # applying the pull term. These will be added to the weights
            # based on position later on and then the exponent will be taken
            WP_this_particle_and_timestep = 0.5*(np.square(pull+force)*(1/model_var)-np.square(force)*(1/model_var))       
            
            # add this onto the weights calculated for this particle at 
            # previous timesteps. This way, when we come to use these weights
            # at an observation timestep, we're using the sum of all weights
            # since the last timestep.            
            W_Proposal[n] += WP_this_particle_and_timestep
            
        #if there are no observations to pull towards then we don't alter the 
        # particle and just set the proposal weights to zero (this happens at 
        # the end of the run after the last observation)
        else:
            W_Proposal[n] = 0.0 



    #If we have an observation here then we need to update our particle weights
    if np.isfinite(Observations[t]):
        print "Observation at "+ str(t) + " is " + str(Observations[t])
        
        # take a record our particles at this stage (for plotting and 
        # re-weighting purposes:
        Particles_before_resampling[:,t] = Particles[:,t]
        
        # calculate the maximum possible weights values. This comes from 
        # solving the qudratic equation which defines the weights due to the 
        # proximity to observation and proximity to f(x(n-1)) for the minimum value
        W_max = 0.5*(np.square(Observations[t] - Particles[:,t]) / (model_var + obs_var))

        # add the log of the proposal weights to the this maximum value
        W_max = W_Proposal[:] + W_max
        
        # Now we move on to the calculate the equivalnet weights:
        # sort these maximum weights into ascending order:
        W_sort = np.sort(W_max)
        
        # define the target weight (particles with max weights below this are 
        # not taken into consideration and are lost at this observation)
        W_target = W_sort[particles_retained*N-1]
        print "particles retained index = " + str(particles_retained*N)
        print "target weight = " + str(W_target)
        print "max weights = " + str(W_sort)
        
        # We use a variable called Kalman Gain (used extensively in Kalman 
        # filters) to help us determine the best position of the particle
        Kalman_gain = model_var / (model_var + obs_var)
        
        # Work through the equations to calculate the position of the individual 
        # particles to achieve this target weight. More detail on these 
        # (including their derivation) can be found in the final references
        for n in range(N):
            
            if W_max[n] <= W_target:
                
                # We use this a lot in the next few steps, so explicitly define the 
                # diference between the observation and the particle
                difference = Observations[t] - Particles[n,t]

                # variable extracted from final equation (breaking things down)
                aaa = Kalman_gain * 0.5 * np.square(difference)/obs_var

                # variable extracted from final equation (breaking things down)
                bbb = 0.5 * np.square(difference)/obs_var - W_target + W_Proposal[n]
                
                # variable extracted from final equation (breaking things down)
                alpha = 1.0 + np.sqrt(1.0 - bbb/aaa + 0.00000001)
                                
                # Calculate final position of the particle
                Particles[n,t] = Particles[n,t] + alpha * Kalman_gain * difference
                
                # Add on a little bit of random error (there are many different 
                # ways of doing this).
                Particles[n,t] += np.sqrt(model_var) * np.random.randn() * 1.e-5
                
            # Calculate (or re-calculate) the difference between the observations 
            # and the particles for these new particle positions
            difference = Observations[t] - Particles[n,t]              
            
            # We will need to know how much the particles have been moved 
            # as a term for the following equations. Calcualte this explicitly now
            movement = Particles[n,t] - Particles_before_resampling[n,t]
            
            # find the additional weights contribution from the distance the             
            # is from the observations and the amount is has been moved, now that
            # we've changed the particle's position. This value should ensure 
            # that the final weights are all roughly equal.
            W_equalising[n,t] = 0.5 * np.square(difference)/obs_var + 0.5 * np.square(movement)/model_var            
            
            # Find the new weights of these particles (they should all be
            # around the same value) 
            Weights[n,t] = W_Proposal[n] + W_equalising[n,t] #DEBUG

        # now we normalise and take the exponent of the log weights
        Weights[:,t] -= np.min(Weights[:,t])
        Weights[:,t] = np.exp(-Weights[:,t])
        Weights[:,t] = Weights[:,t] / np.sum(Weights[:,t])
        
        #print "Weights after obs = " + str(Weights[:,t])

        #######################################################################
        # Re-sampling  ########################################################
        #######################################################################
        # Here we use Stochastic universal sampling, taken from: Kitagawa, G., 
        # 1996: Monte-Carlo filter and smoother for non-Gaussian non-linear 
        # state-space models. Journal of Computational and Graphical 
        # Statistics, 10, 253–259.        
        
        # Arrange the weights into an array so that they are cumulative
        W[0] = Weights[0,t]
        for n in range(1,N):
            W[n] = Weights[n,t] + W[n-1]   
        
        #initialise a random number to act as the first weight interval
        random_interval = np.random.rand()/N
        
        #index used for the re-sampling process.
        index = 0 
        
        # Prepare to step through the arrays of cumulative weights and 
        # particles index by index. 
        for n in range(N):
            
            # if the random interval is greater than the cumulative weights of 
            # the particles at the current array index value then increment to 
            # the next index.(otherwise we will create a duplicate particle as 
            # we sample from this index again)
            while random_interval > W[index]:
                index += 1 
                
            # sample a particle from this array index.
            P[n] = Particles[index,t]

            #increment r by approximately one
            random_interval += 0.9999/N

            # for plotting only -----------------------------------------------
            # update the colours index so that lost particles' colours are 
            # gone and duplicated particles' colour are duplicated
            Colour_index[n,t] = Colour_index[index,t-1]
            
            #add on a little bit of colour noise if this is a duplicate
            if P[n] in P[:n]:
                Colour_index[n,t] += 0.01*np.random.rand()
                
            #------------------------------------------------------------------

        #Re-write Particles[n,t] with the new values
        Particles[:,t] = P
        #print "Particles after resampling = " + str(Particles[:,t])
        
        #Re-set the weights to be even again
        Weights[:,t] = 1.0 / N
        W_Proposal[:] = 0.0 #DEBUG        
        
        #print "Weights after resampling = " + str(Weights[:,t])
        
    # If this isn't an observation timestep the we just update the the weights 
    # with the most recent proposal weight and take the colour indices forward 
    # (for plotting).
    else:
        Log_Weights[:,t] = W_Proposal[:]
       #normalise and take the exponent of the log weights
        Weights[:,t] = np.exp(-Log_Weights[:,t] + np.min(Log_Weights[:,t]))
        Weights[:,t] = Weights[:,t] / np.sum(Weights[:,t])
        #print "Weights after normal = " + str(Weights[:,t])
        
        Colour_index[:,t] = Colour_index[:,t-1]

    #Calculate the ensemble meanand standard deviation
    Particles_mean[t] = np.sum(Weights[:,t]*Particles[:,t])

    
###############################################################################
###############################################################################    
#Plot Everything
###############################################################################
###############################################################################
    
def timeseries_plot(title):
    plt.figure(figsize=(16,6))
    plt.plot(range(Timesteps),Truth,'k',linewidth=10, alpha=0.1, label="Truth")
    plt.plot(range(Timesteps),Particles_mean,'c',linewidth=10, alpha=0.1, label="Particle/Ensemble Mean")
    #plt.plot([0,1],Particles[N/2,0:2],color=plt.cm.winter(float(0)/(N/2)),linewidth=0.5, label="particles")   
    for n in range(N):
        plt.plot(range(0,Timesteps),Particles[n],color='0.9',linewidth=0.5)
    
    for n in range(N):
        for t in range(1,Timesteps):
            if np.isfinite(Observations[t]):
                match = np.where(Colour_index[n,t-1] == Colour_index[:,t])
                try:
                    plt.plot([t-1, t],[Particles[n,t-1],Particles[match[0],t]],color=plt.cm.hsv(Colour_index[n,t-1]),linewidth=1, alpha=Weights[n,t-1]/np.max(Weights[:,t-1]))
                except:
                    plt.plot(t-1,Particles[n,t-1],'x',color=plt.cm.hsv(Colour_index[n,t-1]),markersize=10,linewidth=2)                
            else:
                plt.plot([t-1, t],[Particles[n,t-1], Particles[n,t]],color=plt.cm.hsv(Colour_index[n,t]),linewidth=1, alpha=Weights[n,t-1]/np.max(Weights[:,t-1]))

    plt.errorbar(range(Timesteps),Observations,obs_var*2,fmt='ko',linewidth=2,markersize=8)
    plt.ylabel("Temperature (degC)")
    plt.xlabel("Timestep")
    plt.savefig(title+".png")    
   
        
def moving_hist_with_prob(title):
    fig = plt.figure()
    for t in range(Timesteps):
        fig.clear()
        plt.axes(xlim = (-25,25), ylim=(0,N))
        
        bin_counter=0
       
        #If this is an observation timestep then plotting is much more complicated.      
        if np.isfinite(Observations[t]):
            print "obs plotting"
           
           #plot the observations
            observation = scipy.stats.norm.pdf(np.linspace(-25,25,100),Observations[t],np.sqrt(obs_var))     
            observation = 0.5 * N * observation / np.max(observation)
            plt.fill(np.linspace(-25,25,100),observation,'y', edgecolor="none")
            
            #Find the lost particles
            i_old_remaining = np.in1d(Particles_before_resampling[:,t], Particles[:,t])
            lost = Particles_before_resampling[np.invert(i_old_remaining),t]
            
            #append the two together
            hist = np.sort(np.append(Particles[:,t],lost))
            
            # plot Standard Particles
            for p in range(len(hist)):
                if p>0 and round(hist[p],0)==round(hist[p-1],0):
                    bin_counter+=1
                else:
                    bin_counter=0
                
                if hist[p] in lost:
                    lost_particle_no = np.where(Particles_before_resampling[:,t]==hist[p])[0][0]
                    plt.bar(round(hist[p],0)-0.5,1.0, bottom=bin_counter,width=1.0,facecolor=plt.cm.hsv(Colour_index[lost_particle_no,t-1]), alpha=0.1)
                else:
                    particle_no = np.where(Particles[:,t]==hist[p])[0][0]
                    
                    if hist[p] in hist[:p]:
                        plt.bar(round(hist[p],0)-0.5,1.0, bottom=bin_counter,width=1.0,facecolor=plt.cm.hsv(Colour_index[particle_no,t]), edgecolor="w")
                    else:
                        plt.bar(round(hist[p],0)-0.5,1.0, bottom=bin_counter,width=1.0,facecolor=plt.cm.hsv(Colour_index[particle_no,t]))

        #straightforward non-observation plotting
        else:
            
            hist = np.sort(Particles[:,t])    
            
            # plot Standard Particles
            for n in range(len(hist)):
                if n>0 and round(hist[n],0)==round(hist[n-1],0):
                    bin_counter+=1
                else:
                    bin_counter=0
                    
                particle_no = np.where(Particles[:,t]==hist[n])[0][0]
                plt.bar(round(hist[n],0)-0.5,1.0, bottom=bin_counter,width=1.0,facecolor='0.9')
                plt.bar(round(hist[n],0)-0.5,1.0, bottom=bin_counter,width=1.0,facecolor=plt.cm.hsv(Colour_index[particle_no,t]), alpha=Weights[n,t-1]/np.max(Weights[:,t-1]))
                

        plt.plot(Truth[t],0.75*n,'k*',markersize=20)
        
        plt.title("Timestep = "+str(int(t)))
        plt.xlabel("Temperature (degC)")
        plt.ylabel("Number of particles")
        
        plt.savefig(title+"_"+str(int(t))+".png")


# Call these plotting fuctions
timeseries_plot("Draft_standard_timeseries_"+str(N))
#moving_hist_with_prob("Eq_Weights_hist/Eq_Weights_hist_"+str(N)) 