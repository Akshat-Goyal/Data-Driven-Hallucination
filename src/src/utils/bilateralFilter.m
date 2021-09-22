function output = bilateralFilter(data)
edge = data;

inputHeight = size( data, 1 );
inputWidth = size( data, 2 );
edgeMin = min( edge( : ) );
edgeMax = max( edge( : ) );
edgeDelta = edgeMax - edgeMin;
sigma_domain = min( inputWidth, inputHeight ) / 16;
sigma_range = 0.05 * edgeDelta;
sampling_Domain = sigma_domain;
samplingRange = sigma_range;

derivedSigma_Domain = sigma_domain / sampling_Domain;
derivedSigmaRange = sigma_range / samplingRange;
paddingXY = floor( 2 * derivedSigma_Domain ) + 1;
paddingZ = floor( 2 * derivedSigmaRange ) + 1;
downsampledWidth = floor( ( inputWidth - 1 ) / sampling_Domain ) + 1 + 2 * paddingXY;
downsampledHeight = floor( ( inputHeight - 1 ) / sampling_Domain ) + 1 + 2 * paddingXY;
downsampledDepth = floor( edgeDelta / samplingRange ) + 1 + 2 * paddingZ;
gridData = zeros( downsampledHeight, downsampledWidth, downsampledDepth );
gridWeights = zeros( downsampledHeight, downsampledWidth, downsampledDepth );
[ jj, ii ] = meshgrid( 0 : inputWidth - 1, 0 : inputHeight - 1 );

di = round( ii / sampling_Domain ) + paddingXY + 1;
dj = round( jj / sampling_Domain ) + paddingXY + 1;
dz = round( ( edge - edgeMin ) / samplingRange ) + paddingZ + 1;
for k = 1 : numel(dz),
    dataZ = data(k); % traverses the image column wise, same as di( k )
    if ~isnan(dataZ),
        dik = di(k);
        djk = dj(k);
        dzk = dz(k);
        gridData( dik, djk, dzk ) = gridData( dik, djk, dzk ) + dataZ;
        gridWeights( dik, djk, dzk ) = gridWeights( dik, djk, dzk ) + 1;
    end
end

kernelWidth = 2 * derivedSigma_Domain + 1;
kernelHeight = kernelWidth;
kernelDepth = 2 * derivedSigmaRange + 1;
halfKernelWidth = floor( kernelWidth / 2 );
halfKernelHeight = floor( kernelHeight / 2 );
halfKernelDepth = floor( kernelDepth / 2 );
[gridX, gridY, gridZ] = meshgrid( 0 : kernelWidth - 1, 0 : kernelHeight - 1, 0 : kernelDepth - 1 );
gridX = gridX - halfKernelWidth;
gridY = gridY - halfKernelHeight;
gridZ = gridZ - halfKernelDepth;
gridRSquared = ( gridX .* gridX + gridY .* gridY ) / ( derivedSigma_Domain * derivedSigma_Domain ) + ( gridZ .* gridZ ) / ( derivedSigmaRange * derivedSigmaRange );
kernel = exp( -0.5 * gridRSquared );

blurredGridData = convn( gridData, kernel, 'same' );
blurredGridWeights = convn( gridWeights, kernel, 'same' );
blurredGridWeights( blurredGridWeights == 0 ) = -2;
normalizedBlurredGrid = blurredGridData ./ blurredGridWeights;
normalizedBlurredGrid( blurredGridWeights < -1 ) = 0;
[ jj, ii ] = meshgrid( 0 : inputWidth - 1, 0 : inputHeight - 1 );
di = ( ii / sampling_Domain ) + paddingXY + 1;
dj = ( jj / sampling_Domain ) + paddingXY + 1;
dz = ( edge - edgeMin ) / samplingRange + paddingZ + 1;
output = interpn( normalizedBlurredGrid, di, dj, dz );