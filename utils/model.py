import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import utils_torch
from utils_torch.plot import CreateFigurePlt

def LogSpatialActivity(SpatialActivity, activity, XYs):
    # activity: [BatchSize, StepNum, NeuronNum]
    # XYs: [BatchSize, StepNum, (x, y)]. Can be true XYs or predicted XYs
    BatchSize = activity.shape[0]
    StepNum = activity.shape[1]
    NeuronNum = activity.shape[2]
    XYNum = BatchSize * StepNum

    activity = activity.reshape(XYNum, NeuronNum)
    XYs = XYs.reshape(XYNum, 2)
    ResolutionX, ResolutionY = SpatialActivity.ResolutionX, SpatialActivity.ResolutionY
    
    XYPixels = utils_torch.geometry2D.XYs2PixelIndices(
        XYs, SpatialActivity.BoundaryBox,
        ResolutionX, ResolutionY
    ) # [XYNum, 2]
    
    # Exclude-out-of boundarybox XYs
    Mask1 = XYPixels[:, 0] >= 0
    Mask2 = XYPixels[:, 0] < ResolutionX
    Mask3 = XYPixels[:, 1] >= 0
    Mask4 = XYPixels[:, 1] < ResolutionY
    InsideMasks = Mask1 * Mask2 * Mask3 * Mask4
    XYInsideNum = np.sum(InsideMasks)
    XYOutsideNum = XYNum - XYInsideNum
    utils_torch.AddLog(
        "LogSpatialActivity: %d/%d(%.3f) XYs are outside boundarybox."%(XYOutsideNum, XYNum, XYOutsideNum * 1.0 / XYNum),
        logger="LogSpatialActivity", FileOnly=True
    )
    InsideIndices = np.argwhere(InsideMasks) # [InsideXYsNum, 1]
    InsideIndices = InsideIndices[:, 0]
    
    XYPixels = XYPixels[InsideIndices]
    activity = activity[InsideIndices]

    XYActivitySum = SpatialActivity.XYActivitySum
    XYActivitySquareSum = SpatialActivity.XYActivitySquareSum
    XYActivityCount = SpatialActivity.XYActivityCount
    for XYIndex in range(XYInsideNum):
        # This for-loop might be very time-consuming, but there does not seem a way to do it using numpy array indexing.
        XY = XYPixels[XYIndex]
        X, Y = XY[0], XY[1]
        XYActivitySum[X, Y] += activity[XYIndex]
        XYActivitySquareSum[X, Y] += activity[XYIndex]
        XYActivityCount[X, Y] += 1

def InitSpatialActivity(BoundaryBox, Resolution, NeuronNum):
    SpatialActivity = utils_torch.PyObj()
    SpatialActivity.BoundaryBox = utils_torch.plot.CopyBoundaryBox(BoundaryBox)
    ResolutionX, ResolutionY = utils_torch.plot.ParseResolution(BoundaryBox.Width, BoundaryBox.Height, Resolution)
    SpatialActivity.ResolutionX = ResolutionX
    SpatialActivity.ResolutionY = ResolutionY
    SpatialActivity.NeuronNum = NeuronNum

    SpatialActivity.XYActivitySum = np.zeros((ResolutionX, ResolutionY, NeuronNum), dtype=np.float32)
    SpatialActivity.XYActivitySquareSum = np.zeros((ResolutionX, ResolutionY, NeuronNum), dtype=np.float32)
    SpatialActivity.XYActivityCount = np.zeros((ResolutionX, ResolutionY), dtype=np.int32)
    
    return SpatialActivity

def ClearSpatialActivity(SpatialActivity):
    ResolutionX, ResolutionY, NeuronNum = SpatialActivity.ResolutionX, SpatialActivity.ResolutionY, SpatialActivity.NeuronNum
    SpatialActivity.XYActivitySum = np.zeros((ResolutionX, ResolutionY, NeuronNum), dtype=np.float32)
    SpatialActivity.XYActivitySquareSum = np.zeros((ResolutionX, ResolutionY, NeuronNum), dtype=np.float32)
    SpatialActivity.XYActivityCount = np.zeros((ResolutionX, ResolutionY), dtype=np.int32)

def CalculateSpatialActivity(SpatialActivity):
    XYActivitCount = SpatialActivity.XYActivityCount[:, :, np.newaxis]
    SpatialActivity.XYActivityMean = SpatialActivity.XYActivitySum / XYActivitCount
    # DX = E(X^2) - (EX)^2
    SpatialActivity.XYActivityVar = SpatialActivity.XYActivitySquareSum / XYActivitCount - SpatialActivity.XYActivityMean ** 2
    SpatialActivity.XYActivityStd = SpatialActivity.XYActivityVar ** 0.5

def PlotSpatialActivity(
        SpatialActivity, Arena, PageSize=100,
        SaveDir=None, SaveName="", **kw
    ):

    NeuronName = kw.setdefault("NeuronName", "Neuron")

    ActivityMean = SpatialActivity.XYActivityMean.transpose(2, 0, 1) # [NeuronNum, ResolutionX, ResolutionY]
    ActivityStd = SpatialActivity.XYActivityStd.transpose(2, 0, 1) # [NeuronNum, ResolutionX, ResolutionY]
    ActivitySampleNum = SpatialActivity.XYActivityCount # [ResolutionX, ResolutionY]

    ResolutionX, ResolutionY, NeuronNum = SpatialActivity.ResolutionX, SpatialActivity.ResolutionY, SpatialActivity.NeuronNum

    PlotNum = NeuronNum # To be extended.
    PlotIndices = range(NeuronNum)


    ActivityMeanStat = utils_torch.math.NpStatistics(ActivityMean)
    ActivityStdStat = utils_torch.math.NpStatistics(ActivityStd)

    ActivityMeanColorRange = [
        max(ActivityMeanStat.Min, ActivityMeanStat.Mean - 5.0 * ActivityMeanStat.Std),
        min(ActivityMeanStat.Max, ActivityMeanStat.Mean + 5.0 * ActivityMeanStat.Std)
    ]
    ActivityStdColorRange = [
        max(ActivityStdStat.Min, ActivityStdStat.Mean - 5.0 * ActivityStdStat.Std),
        min(ActivityStdStat.Max, ActivityStdStat.Mean + 5.0 * ActivityStdStat.Std)
    ]

    ActivityColorRange = [
        min(ActivityMeanColorRange[0], ActivityStdColorRange[0]),
        max(ActivityMeanColorRange[1], ActivityStdColorRange[1])
    ]

    ActivityMeanColored = utils_torch.plot.Map2Color(
        ActivityMean, Method="GivenMinMax", Min=ActivityColorRange[0], Max=ActivityColorRange[1]
    )
    ActivityStdColored = utils_torch.plot.Map2Color(
        ActivityStd, Method="GivenMinMax", Min=ActivityColorRange[0], Max=ActivityColorRange[1]
    )

    BoundaryBox = Arena.GetBoundaryBox()
    XTicks, XTicksStr = utils_torch.plot.CalculateTicksFloat(Min=BoundaryBox.XMin, Max=BoundaryBox.XMax)
    YTicks, YTicksStr = utils_torch.plot.CalculateTicksFloat(Min=BoundaryBox.YMin, Max=BoundaryBox.YMax)
    Ticks = {
        "XTicks": XTicks,
        "XTicksStr": XTicksStr,
        "YTicks": YTicks,
        "YTicksStr": YTicksStr,
    }
    
    PageNum = PlotNum // PageSize
    if PlotNum % PageSize > 0:
        PageNum += 1
    for PageIndex in range(PageNum):
        PlotIndexStart = PageIndex * PageNum
        PlotIndexEnd = min((PageIndex + 1) * PageSize, NeuronNum)
        PlotNum = PlotIndexEnd - PlotIndexStart
        PlotIndicesSub = PlotIndices[PlotIndexStart:PlotIndexEnd]
        fig, axes = utils_torch.plot.CreateFigurePlt(PlotNum=PageSize)
        for AxIndex, PlotIndex in zip(range(PlotNum), PlotIndicesSub):
            ax = utils_torch.plot.GetAx(axes, AxIndex)
            utils_torch.plot.PlotMatrix(
                ax, ActivityMeanColored.dataColored[PlotIndex, :, :, :], IsDataColored=True,
                XYRange=BoundaryBox, Ticks=Ticks, Title="%s-No.%d"%(NeuronName, PlotIndex)
            )
        plt.suptitle(SaveName + "Mean-Neurons%d~%d.png")
        utils_torch.plot.SaveFigForPlt(
            SavePath = SaveDir + SaveName + "-Mean-Neurons%d~%d.png"%(PlotIndexStart, PlotIndexEnd)
        )

    for PageIndex in range(PageNum):
        PlotIndexStart = PageIndex * PageNum
        PlotIndexEnd = min((PageIndex + 1) * PageSize, NeuronNum)
        PlotNum = PlotIndexEnd - PlotIndexStart
        PlotIndicesSub = PlotIndices[PlotIndexStart:PlotIndexEnd]
        fig, axes = utils_torch.plot.CreateFigurePlt(PlotNum=PageSize)
        for AxIndex, PlotIndex in zip(PlotIndicesSub, range(PlotNum)):
            ax = utils_torch.plot.GetAx(axes, AxIndex)
            utils_torch.plot.PlotMatrix(
                ax, ActivityStdColored.dataColored[PlotIndex, :, :, :], IsDataColored=True,
                XYRange=BoundaryBox, Ticks=Ticks, Title="%s-No.%d"%(NeuronName, PlotIndex)
            )
        plt.suptitle(SaveName + "-Std-Neurons%d~%d.png")
        utils_torch.plot.SaveFigForPlt(
            SavePath = SaveDir + SaveName + "-Std-Neurons%d~%d.png"%(PlotIndexStart, PlotIndexEnd)
        )
    
    fig, axes = utils_torch.plot.CreateFigurePlt(6, RowNum=2, ColNum=3)
    ax = utils_torch.plot.GetAx(axes, 0)
    utils_torch.plot.PlotColorBarInSubAx(
        ax, Location=[0.4, 0.0, 0.2, 1.0],
        Min=ActivityColorRange[0], Max=ActivityColorRange[1]
    )
    ax.axis("off")
    ax.set_title("%s Mean Color Map"%SaveName)

    ax = utils_torch.plot.GetAx(axes, 1)
    utils_torch.plot.PlotHistogram(
        ax, data=ActivityMean,  XLabel="%s Mean"%SaveName, YLabel="Proportion"
    )

    ax = utils_torch.plot.GetAx(axes, 3)
    utils_torch.plot.PlotColorBarInSubAx(
        ax, Location=[0.4, 0.0, 0.2, 1.0], 
        Min=ActivityColorRange[0], Max=ActivityColorRange[1]
    )
    ax.axis("off")
    ax.set_title("%s Std Color Map"%SaveName)

    ax = utils_torch.plot.GetAx(axes, 4)
    utils_torch.plot.PlotHistogram(
        ax, data=ActivityStd, XLabel="%s Std"%SaveName, YLabel="Proportion"
    )

    ax = utils_torch.plot.GetAx(axes, 2)
    utils_torch.plot.PlotMatrixWithColorBar(
        ax, ActivitySampleNum, IsDataColored=False,
        XYRange=BoundaryBox, Ticks=Ticks, Title="Sample Num Distribution"
    )
    ax.set_title("Sample Num Spatial Distribution")

    utils_torch.plot.SaveFigForPlt(
        SavePath = SaveDir + SaveName + "Std-Stats.svg"
    )