#define FRACT_BITS_J 8
#define FRACT_BITS_ERROR 24
#define FIXED_ONE ( 1 << FRACT_BITS )
#define INT2FIXED(x, q) ( (x) << q )
#define FLOAT2FIXED(x, q) ( (long)((x) * (1L << q)) )
#define FIXED2INT(x, q) ( (x) >> q )
#define FIXED2FLOAT(x, q) ( ((float)(x)) / (1L << q) )
#define MULT(x, y, q) ( ((x >> q/2) * (y >> q/2)) )

typedef struct sTrackDataFixedPoint {
	int result;
	int error;
	int J[6];
} TrackDataFixedPoint;