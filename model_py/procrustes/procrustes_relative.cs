using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

public class ProcrustesRel
{
    private readonly Matrix<float> reference;

    public ProcrustesRel(Matrix<float> refFace)
    {
        if (refFace == null)
            throw new ArgumentException("Reference tensor cannot be null.");

        var refCentroid = refFace.ColumnSums() / refFace.RowCount;
        var refCentered = refFace - DenseMatrix.OfRowVectors(Enumerable.Repeat(refCentroid, refFace.RowCount));
        float refNorm = (float)refCentered.FrobeniusNorm();
        this.reference = refCentered / refNorm;
    }

    public Matrix<float> Forward(Matrix<float> x)
    {
        if (x == null)
            throw new ArgumentException("Input matrix cannot be null.");
        if (x.RowCount != reference.RowCount || x.ColumnCount != reference.ColumnCount)
            throw new ArgumentException("Input must have same dimensions as reference.");

        var xCentroid = x.ColumnSums() / x.RowCount;
        var xCentered = x - DenseMatrix.OfRowVectors(Enumerable.Repeat(xCentroid, x.RowCount));
        float xNorm = (float)xCentered.FrobeniusNorm();
        var xScaled = xCentered / xNorm;

        var product = xScaled.TransposeThisAndMultiply(reference);
        var svd = product.Svd(computeVectors: true);
        var U = svd.U;
        var Vt = svd.VT;

        var R = U * Vt;
        var xAligned = xScaled * R;
        return xAligned;
    }

}
