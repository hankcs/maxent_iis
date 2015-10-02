
/**
 * ÊµÀı
 */
public class Instance
{
    /**
     * ±êÇ©
     */
    public int label;
    /**
     * ÌØÕ÷
     */
    public Feature feature;

    public Instance(int label, int[] xs)
    {
        this.label = label;
        this.feature = new Feature(xs);
    }

    public int getLabel()
    {
        return label;
    }

    public Feature getFeature()
    {
        return feature;
    }

    @Override
    public String toString()
    {
        return "Instance{" +
                "label=" + label +
                ", feature=" + feature +
                '}';
    }
}
