import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pdb, copy



##### PARAMETERS #######
rows_in_plot = 1
columns_in_plot = 4
standard_time = 1000
standard_max = 1
mean_window = 50
median_window = 50
data_types = ['step', 'impulse', 'ramp', 'sin']
plt_titles = {'step': 'Step Function',
              'impulse': 'Impulse Function',
              'ramp': 'Ramp Function',
              'sin': 'Sine Function',
              }
              
sns.set(style="white", palette="bright")


class SimpleData:
    def __init__(self, filtering = 0, ftype = None):
        self.data = None
        self.filtering = filtering
        # 0 is no noise
        # 1 is noise
        # 2 is mean temporal
        # 3 is median temporal
        self.plot1 = plt.figure()
        self.plot_location = 1
        self.mean_window = mean_window
        self.median_window = median_window

##        fig = pyplot.gcf()
##        fig.canvas.set_window_title('My title')
        
        self.run_loop(ftype)

        
    def run_loop(self, ftype):
        self.funclist = {'step'   : self.step,
                         'impulse': self.impulse,
                         'ramp'   : self.ramp,
                         'sin'    : self.sin,
                         }
        if ftype != None:
            loop_through = [self.funclist[ftype]]
        else:
            loop_through = self.funclist.items()

        for ftype,every in loop_through:
            every()
            if self.filtering >= 1:
                self.add_noise()
            if self.filtering == 2:
                self.temporal_mean()
            elif self.filtering == 3:
                self.temporal_median()
            
            self.type = ftype
            self.plot_data()
        

    def step(self, x = standard_time, y = standard_max):
        data = [(i,0) if i < x else (i,y) for i in range(2*x)]
        self.data = np.array(data).astype(float)

    def impulse(self, x = standard_time, y = standard_max):
        data = [(i,y) if i == x else (i,0) for i in range(2*x)]
        self.data = np.array(data).astype(float)

    def ramp(self, x = standard_time, y = standard_max):
        data = [(i,i/(2*x)*y) for i in range(2*x)]
        self.data = np.array(data).astype(float)

    def sin(self, x = standard_time, y = standard_max):
        data = [(i,np.sin(i/x*4)) for i in range(2*x)]
        self.data = np.array(data).astype(float)

    def add_noise(self, mean = 0, dev = 0.1):
        dims = self.data.shape
        noise = np.random.normal(mean, dev, dims[0])
        self.data[:,1] += noise

    def temporal_mean(self):
        smoothed = list()
        for i in range(len(self.data)-self.mean_window):
            smoothed.append(np.mean(self.data[i:i+self.mean_window], axis = 0))
        
        self.data = np.array(smoothed)

    def temporal_median(self):
        smoothed = list()
        for i in range(len(self.data)-self.mean_window):
            smoothed.append(np.median(self.data[i:i+self.median_window], axis = 0))

        self.data = np.array(smoothed)

    def plot_data(self):
        x = self.data[:,0]
        y = self.data[:,1]
        self.ax1 = self.plot1.add_subplot(rows_in_plot,columns_in_plot,self.plot_location)
        self.ax1.plot(x, y, 'b-')
        plt.title(plt_titles[self.type])
        plt.xlabel('Time')
        plt.ylim([-1,1.2])
        if self.plot_location == 1:
            plt.ylabel('Reading')
        self.plot_location += 1
        

##    def
    
    def show_plot(self):
        plt.show()
        

# Part 1: simple data
simple = SimpleData(0)

# Part 2: Add Noise
noise = SimpleData(1)

# Part 3: Mean Temporal
MT = SimpleData(2)

#Part 4: Median Temporal
MdT = SimpleData(3)


plt.show()
