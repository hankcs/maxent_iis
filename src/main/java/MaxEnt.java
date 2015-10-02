import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;

/**
 * 最大熵的IIS（Improved Iterative Scaling）训练算法
 * User: tpeng  <pengtaoo@gmail.com>
 */
public class MaxEnt
{

    private final static boolean DEBUG = false;

    /**
     * 迭代次数
     */
    private final int ITERATIONS = 200;

    /**
     * 浮点数精度
     */
    private static final double EPSILON = 0.001;

    // the number of training instances
    /**
     * 训练实例数
     */
    private int N;

    // the minimal of Y
    /**
     * Y的最小值
     */
    private int minY;

    // the maximum of Y
    /**
     * Y的最大值
     */
    private int maxY;

    // the empirical expectation value of f(x, y)
    /**
     * 联合分布的期望
     */
    private double empirical_expects[];

    // the weight to learn.
    /**
     * 模型参数
     */
    private double w[];

    /**
     * 实例列表
     */
    private List<Instance> instances = new ArrayList<Instance>();

    /**
     * 特征函数列表
     */
    private List<FeatureFunction> functions = new ArrayList<FeatureFunction>();

    /**
     * 特征列表
     */
    private List<Feature> features = new ArrayList<Feature>();

    public static void main(String... args) throws FileNotFoundException
    {
        List<Instance> instances = DataSet.readDataSet("examples/zoo.train");
        MaxEnt me = new MaxEnt(instances);
        me.train();

        List<Instance> trainInstances = DataSet.readDataSet("examples/zoo.test");
        int pass = 0;
        for (Instance instance : trainInstances)
        {
            int predict = me.classify(instance);
            if (predict == instance.getLabel())
            {
                pass += 1;
            }
        }

        System.out.println("accuracy: " + 1.0 * pass / trainInstances.size());
    }

    public MaxEnt(List<Instance> trainInstance)
    {

        instances.addAll(trainInstance);
        N = instances.size();
        createFeatFunctions(instances);
        w = new double[functions.size()];
        empirical_expects = new double[functions.size()];
        calc_empirical_expects();
    }

    /**
     * 创建特征函数
     * @param instances 实例
     */
    private void createFeatFunctions(List<Instance> instances)
    {

        int maxLabel = 0;
        int minLabel = Integer.MAX_VALUE;
        int[] maxFeatures = new int[instances.get(0).getFeature().getValues().length];
        LinkedHashSet<Feature> featureSet = new LinkedHashSet<Feature>();

        for (Instance instance : instances)
        {

            if (instance.getLabel() > maxLabel)
            {
                maxLabel = instance.getLabel();
            }
            if (instance.getLabel() < minLabel)
            {
                minLabel = instance.getLabel();
            }


            for (int i = 0; i < instance.getFeature().getValues().length; i++)
            {
                if (instance.getFeature().getValues()[i] > maxFeatures[i])
                {
                    maxFeatures[i] = instance.getFeature().getValues()[i];
                }
            }

            featureSet.add(instance.getFeature());
        }

        features = new ArrayList<Feature>(featureSet);

        maxY = maxLabel;
        minY = minLabel;

        for (int i = 0; i < maxFeatures.length; i++)
        {
            for (int x = 0; x <= maxFeatures[i]; x++)
            {
                for (int y = minY; y <= maxLabel; y++)
                {
                    functions.add(new FeatureFunction(i, x, y));
                }
            }
        }

        if (DEBUG)
        {
            System.out.println("# features = " + features.size());
            System.out.println("# functions = " + functions.size());
        }
    }

    // calculates the p(y|x)
    /**
     * 计算条件概率 p(y|x)
     * @return p(y|x)
     */
    private double[][] calc_prob_y_given_x()
    {

        double[][] cond_prob = new double[features.size()][maxY + 1];

        for (int y = minY; y <= maxY; y++)
        {
            for (int i = 0; i < features.size(); i++)
            {
                double z = 0;
                for (int j = 0; j < functions.size(); j++)
                {
                    z += w[j] * functions.get(j).apply(features.get(i), y);
                }
                cond_prob[i][y] = Math.exp(z);
            }
        }

        for (int i = 0; i < features.size(); i++)
        {
            double normalize = 0;
            for (int y = minY; y <= maxY; y++)
            {
                normalize += cond_prob[i][y];
            }
            for (int y = minY; y <= maxY; y++)
            {
                cond_prob[i][y] /= normalize;
            }
        }

        return cond_prob;
    }

    /**
     * 训练
     */
    public void train()
    {
        for (int k = 0; k < ITERATIONS; k++)
        {
            for (int i = 0; i < functions.size(); i++)
            {
                double delta = iis_solve_delta(empirical_expects[i], i);
                w[i] += delta;
            }
            if (DEBUG)  System.out.println("ITERATIONS: " + k + " " + Arrays.toString(w));
        }
    }

    /**
     * 分类
     * @param instance
     * @return
     */
    public int classify(Instance instance)
    {

        double max = 0;
        int label = 0;

        for (int y = minY; y <= maxY; y++)
        {
            double sum = 0;
            for (int i = 0; i < functions.size(); i++)
            {
                sum += Math.exp(w[i] * functions.get(i).apply(instance.getFeature(), y));
            }
            if (sum > max)
            {
                max = sum;
                label = y;
            }
        }
        return label;
    }

    /**
     * 计算经验期望
     */
    private void calc_empirical_expects()
    {

        for (Instance instance : instances)
        {
            int y = instance.getLabel();
            Feature feature = instance.getFeature();
            for (int i = 0; i < functions.size(); i++)
            {
                empirical_expects[i] += functions.get(i).apply(feature, y);
            }
        }
        for (int i = 0; i < functions.size(); i++)
        {
            empirical_expects[i] /= 1.0 * N;
        }
        if (DEBUG)  System.out.println(Arrays.toString(empirical_expects));
    }

    /**
     * 命中的所有特征函数输出之和
     * @param feature
     * @param y
     * @return
     */
    private int apply_f_sharp(Feature feature, int y)
    {

        int sum = 0;
        for (int i = 0; i < functions.size(); i++)
        {
            FeatureFunction function = functions.get(i);
            sum += function.apply(feature, y);
        }
        return sum;
    }

    /**
     * 求delta_i
     * @param empirical_e fi的期望
     * @param fi fi的下标
     * @return delta_i
     */
    private double iis_solve_delta(double empirical_e, int fi)
    {

        double delta = 0;
        double f_newton, df_newton;
        double p_yx[][] = calc_prob_y_given_x();

        int iters = 0;

        while (iters < 50)                                  // 牛顿法
        {
            f_newton = df_newton = 0;
            for (int i = 0; i < instances.size(); i++)
            {
                Instance instance = instances.get(i);
                Feature feature = instance.getFeature();
                int index = features.indexOf(feature);
                for (int y = minY; y <= maxY; y++)
                {
                    int f_sharp = apply_f_sharp(feature, y);
                    double prod = p_yx[index][y] * functions.get(fi).apply(feature, y) * Math.exp(delta * f_sharp);
                    f_newton += prod;
                    df_newton += prod * f_sharp;
                }
            }
            f_newton = empirical_e - f_newton / N;      // g
            df_newton = -df_newton / N;                 // g的导数

            if (Math.abs(f_newton) < 0.0000001)
                return delta;

            double ratio = f_newton / df_newton;

            delta -= ratio;
            if (Math.abs(ratio) < EPSILON)
            {
                return delta;
            }
            iters++;
        }
        throw new RuntimeException("IIS did not converge"); // w_i不收敛
    }

    /**
     * 特征函数
     */
    class FeatureFunction
    {

        private int index;
        private int value;
        private int label;

        FeatureFunction(int index, int value, int label)
        {
            this.index = index;
            this.value = value;
            this.label = label;
        }

        /**
         * 代入函数
         * @param feature 特征X（维度由构造时的index指定）
         * @param label Y
         * @return
         */
        public int apply(Feature feature, int label)
        {
            if (feature.getValues()[index] == value && label == this.label)
                return 1;
            return 0;
        }
    }
}


