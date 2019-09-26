/* 
 * This file implements the time evolution of quantities relevant to single 
 * stellar populations (SSPs). In the current version of VICE, this includes 
 * the single stellar population enrichment function, the main sequence mass 
 * fraction, and the cumulative return fraction. In general, these features 
 * ease the computational expense of singlezone simulations. 
 * 
 * NOTES 
 * ===== 
 * There are factors of 0.08 and 0.04 on the Kroupa IMF for the 0.08 Msun < 
 * m < 0.5 Msun and m > 0.5 Msun mass ranges. This ensures that the IMF 
 * returns the same values on either side of 0.08 and 0.5 Msun. These factors 
 * are independent of the normalization of the IMF and must appear for the 
 * distribution of stellar masses to be consistent. 
 */ 

#include <stdlib.h> 
#include <string.h> 
#include <math.h> 
#include "agb.h" 
#include "imf.h" 
#include "ccsne.h" 
#include "sneia.h"  
#include "ssp.h" 
#include "singlezone.h" 
#include "quadrature.h" 

/* ---------- static function comment headers not duplicated here ---------- */
static double Kalirai08_remnant_mass(double m); 
static double CRFnumerator_integrand(double m); 
static double CRFdenominator_integrand(double m); 
static double CRFnumerator_Kalirai08(SSP ssp, double time); 
static double CRFnumerator_Kalirai08_IMFrange(double m_upper, 
	double turnoff_mass, double m_lower, double a); 
static double CRFnumerator_Kalirai08_above_8Msun(double m_upper, 
	double turnoff_mass, double a); 
static double CRFnumerator_Kalirai08_below_8Msun(double m_upper, 
	double turnoff_mass, double a); 
static double CRFdenominator(SSP ssp); 
static double CRFdenominator_IMFrange(double m_upper, double m_lower, 
	double a); 
static double MSMFdenominator(SSP ssp); 
static double MSMFnumerator(SSP ssp, double time); 
static IMF_ *ADOPTED_IMF = NULL; 

/* 
 * Allocate memory for and return a pointer to an SSP struct. Automatically 
 * sets both crf and msmf to NULL. Allocates memory for a 100-element char * 
 * IMF specifier. 
 * 
 * header: ssp.h 
 */ 
extern SSP *ssp_initialize(void) { 

	/* 
	 * The imf object is initialized with the default lower and upper mass 
	 * limits. They will be modified by the python wrapper in VICE anyway. 
	 * This keeps the implementation simpler in that now singlezone_initialize 
	 * does not need to take arguments for the mass limits. 
	 */ 

	SSP *ssp = (SSP *) malloc (sizeof(SSP)); 
	ssp -> imf = imf_initialize(0.08, 100); 
	ssp -> crf = NULL; 
	ssp -> msmf = NULL; 
	return ssp; 

} 

/* 
 * Free up the memory stored in an SSP struct. 
 * 
 * header: ssp.h 
 */ 
extern void ssp_free(SSP *ssp) { 

	if (ssp != NULL) {

		if ((*ssp).crf != NULL) {
			free(ssp -> crf); 
			ssp -> crf = NULL; 
		} else {} 

		if ((*ssp).msmf != NULL) {
			free(ssp -> msmf); 
			ssp -> msmf = NULL; 
		} else {} 

		if ((*ssp).imf != NULL) {
			imf_free(ssp -> imf); 
			ssp -> imf = NULL; 
		} else {} 

		free(ssp); 
		ssp = NULL; 

	} else {}  

} 

/* 
 * Determine the main sequence turnoff mass in solar masses of a single 
 * stellar population at a time t in Gyr following their formation. 
 * 
 * Parameters 
 * ========== 
 * t: 			Time in Gyr 
 * postMS: 		Ratio of a star's post main sequence lifetime to its main 
 * 				sequence lifetime 
 * 
 * Returns 
 * ======= 
 * Main sequence turnoff mass in solar masses via 
 * (t / (1 + postMS)(10 Gyr))^(-1/3.5) 
 * 
 * Notes 
 * ===== 
 * Versions >= 1.1: This is the mass of a dying star taking into account their 
 * 		main sequence lifetimes. 
 * 10 Gyr and 3.5 are values that can be changed in ssp.h  
 * 
 * header: utils.h 
 */ 
extern double main_sequence_turnoff_mass(double t, double postMS) { 

	/* m_to = (t / ((1 + postMS) * 10 Gyr))^(-1/3.5) */ 
	return pow( 
		t / ((1 + postMS) * SOLAR_LIFETIME), 
		-1.0 / MASS_LIFETIME_PLAW_INDEX
	); 

} 

/* 
 * The Kalirai et al. (2008) initial-final remnant mass relationship 
 * 
 * Parameters 
 * ========== 
 * m: 		The initial stellar mass in Msun 
 * 
 * Returns 
 * ======= 
 * The mass of the remnant under the Kalirai et al. (2008) model. Stars with 
 * main sequence masses >= 8 Msun leave behind a 1.44 Msun remnant. Those < 8 
 * Msun leave behind a 0.394 + 0.109 * m Msun mass remnant. 
 * 
 * References 
 * ========== 
 * Kalirai et al. (2008), ApJ, 676, 594 
 */ 
static double Kalirai08_remnant_mass(double m) {

	if (m >= 8) {
		return 1.44; 
	} else if (0 < m && m < 8) {
		return 0.394 + 0.109 * m; 
	} else {
		return 0; 
	} 

} 

/* 
 * The integrand of the numerator of the cumulative return fraction (CRF). 
 * 
 * Parameters 
 * ========== 
 * m: 		The initial stellar mass in Msun 
 * 
 * Returns 
 * ======= 
 * The difference in the initial stellar mass and remnant mass weighted by 
 * the adopted IMF 
 * 
 * See Also 
 * ======== 
 * Section 2.2 of Science Documentation: The Cumulative Return Fraction 
 */ 
static double CRFnumerator_integrand(double m) { 

	return (m - Kalirai08_remnant_mass(m)) * imf_evaluate(*ADOPTED_IMF, m); 

} 

/* 
 * The integrand of the denominator of the cumulative return fraction (CRF). 
 * 
 * Parameters 
 * ========== 
 * m: 		The initial stellar mass in Msun 
 * 
 * Returns 
 * ======= 
 * The initial stellar mass weighted by the adopted IMF 
 * 
 * See Also 
 * ======== 
 * Section 2.2 of Science Documentation: The Cumulative Return Fraction 
 */ 
static double CRFdenominator_integrand(double m) {

	return m * imf_evaluate(*ADOPTED_IMF, m); 

}

/* 
 * Run a simulation of elemental production for a single element produced by a 
 * single stellar population. 
 * 
 * Parameters 
 * ========== 
 * ssp: 		A pointer to an SSP object 
 * e: 			A pointer to an element to run the simulation for 
 * Z: 			The metallicity by mass of the stellar population 
 * times: 		The times at which the simulation will evaluate 
 * n_times: 	The number of elements in the times array 
 * mstar: 		The mass of the stellar population in Msun 
 * 
 * Returns 
 * ======= 
 * An array of the same length as times, where each element is the mass of the 
 * given chemical element at the corresponding time. NULL on failure to 
 * allocate memory. 
 * 
 * header: ssp.h 
 */ 
extern double *single_population_enrichment(SSP *ssp, ELEMENT *e, 
	double Z, double *times, unsigned long n_times, double mstar) {

	double *mass = (double *) malloc (n_times * sizeof(double)); 
	if (mass == NULL) return NULL; 	/* memory error */ 

	ssp -> msmf = (double *) malloc (n_times * sizeof(double)); 
	if ((*ssp).msmf == NULL) return NULL; /* memory error */ 
	double denominator = MSMFdenominator(*ssp); 
	if (denominator < 0) { /* unrecognized IMF */ 
		free(mass); 
		free(ssp -> msmf); 
		return NULL; 
	} else {
		unsigned long i; 
		for (i = 0l; i < n_times; i++) {
			ssp -> msmf[i] = MSMFnumerator(*ssp, times[i]) / denominator; 
		}
	} 

	mass[0] = 0; 
	if (n_times >= 2l) { 
		/* The contribution from CCSNe */ 
		mass[1] = get_cc_yield(*e, Z) * mstar; 
		unsigned long i; 
		for (i = 2l; i < n_times; i++) {
			mass[i] = mass[i - 1l]; 		/* previous timesteps */ 

			/* The contribution from SNe Ia */ 
			mass[i] += ((*(*e).sneia_yields).yield_ * 
				(*(*e).sneia_yields).RIa[i] * mstar); 

			/* The contribution from AGB stars */ 
			mass[i] += (
				get_AGB_yield(*e, Z, 
					main_sequence_turnoff_mass(times[i], (*ssp).postMS)) * 
				mstar * ((*ssp).msmf[i] - (*ssp).msmf[i + 1l])
			); 

		}
	} else {} 

	return mass; 

} 

/* 
 * Determine the mass recycled from all previous generations of stars for 
 * either a given element or the gas supply. For details, see section 3.3 of 
 * VICE's science documentation. 
 * 
 * Parameters 
 * ========== 
 * sz: 		The singlezone object for the current simulation 
 * e: 		A pointer to the element to find the recycled mass. NULL to find 
 * 			it for the total ISM gas. 
 * 
 * Returns 
 * ======= 
 * The recycled mass in Msun 
 * 
 * header: ssp.h 
 */ 
extern double mass_recycled(SINGLEZONE sz, ELEMENT *e) {

	/* ----------------------- Continuous recycling ----------------------- */ 
	if ((*sz.ssp).continuous) {
		unsigned long i; 
		double mass = 0; 
		/* From each previous timestep, there's a dCRF contribution */ 
		for (i = 0l; i <= sz.timestep; i++) {
			if (e == NULL) { 		/* This is the gas supply */ 
				mass += ((*sz.ism).star_formation_history[sz.timestep - i] * 
					sz.dt * ((*sz.ssp).crf[i + 1l] - (*sz.ssp).crf[i])); 
			} else { 			/* element -> weight by Z */ 
				mass += ((*sz.ism).star_formation_history[sz.timestep - i] * 
					sz.dt * ((*sz.ssp).crf[i + 1l] - (*sz.ssp).crf[i]) * 
					(*e).Z[sz.timestep - i]); 
			} 
		} 
		return mass; 
	/* ---------------------- Instantaneous recycling ---------------------- */ 
	} else {
		if (e == NULL) {			/* gas supply */ 
			return (*sz.ism).star_formation_rate * sz.dt * (*sz.ssp).R0; 
		} else { 				/* element -> weight by Z */ 
			return ((*sz.ism).star_formation_rate * sz.dt * (*sz.ssp).R0 * 
				(*e).mass / (*sz.ism).mass); 
		}
	}

} 

/* 
 * Re-enriches each zone in a multizone simulation. Zones with instantaneous 
 * recycling will behave as such, but zones with continuous recycling will 
 * produce tracer particles that re-enrich their current zone, even if that 
 * zone has instantaneous recycling. 
 * 
 * Parameters 
 * ========== 
 * mz: 		A pointer to the multizone object to re-enrich 
 * 
 * header: ssp.h 
 */ 
extern void recycle_metals_from_tracers(MULTIZONE *mz, unsigned int index) { 

	/* 
	 * Look at each tracer particle and allow each that was born in a zone 
	 * with continuous recycling to enrich its current zone via continuous 
	 * recycling, regardless of the current zone's recycling prescription. 
	 * Zones that have instantaneous recycling will retain their recycling 
	 * as such as well as that from particles with continuous recycling that 
	 * migrate into that zone. 
	 */ 

	unsigned long i; 
	for (i = 0l; i < (*(*mz).mig).tracer_count; i++) {
		TRACER *t = mz -> mig -> tracers[i]; 
		SSP *ssp = mz -> zones[(*t).zone_origin] -> ssp; 

		if ((*ssp).continuous) { 
			/* ------------------- Continuous recycling ------------------- */ 
			unsigned long n = (*(*mz).zones[0]).timestep - (*t).timestep_origin; 
			/* The metallicity by mass of this element in the tracer */ 
			double Z = (
				(*(*(*mz).zones[(*t).zone_origin]).elements[index]).Z[(
					*t).timestep_origin] 
			); 
			mz -> zones[(*t).zone_current] -> elements[index] -> mass += (
				Z * (*t).mass * ((*ssp).crf[n + 1l] - (*ssp).crf[n])
			); 
		} else {}

	} 

	unsigned int j; 
	for (j = 0; j < (*(*mz).mig).n_zones; j++) {
		SSP *ssp = mz -> zones[j] -> ssp; 

		if (!(*ssp).continuous) {
			/* ------------------ Instantaneous recycling ------------------ */ 
			mz -> zones[j] -> elements[index] -> mass += (
				(*(*(*mz).zones[j]).ism).star_formation_rate * 
				(*(*mz).zones[j]).dt * 
				(*(*(*mz).zones[j]).ssp).R0 * 
				(*(*(*mz).zones[j]).elements[index]).mass / 
				(*(*(*mz).zones[j]).ism).mass 
			); 
		} else {} 

	}

} 

/* 
 * Determine the amount of ISM gas recycled from stars in each zone in a 
 * multizone simulation. Just as is the case with re-enrichment of metals, 
 * zones with instantaneous recycling will behave as such, but zones with 
 * continuous recycling will produce tracer particles that re-enrich their 
 * current zone, even if that zone has instantaneous recycling. 
 * 
 * Parameters 
 * ========== 
 * mz: 		The multizone object for this simulation 
 * 
 * Returns 
 * ======= 
 * An array of doubles, each element is the mass in Msun of ISM gas returned 
 * to each zone at the current timestep. 
 * 
 * header: ssp.h 
 */ 
extern double *gas_recycled_in_zones(MULTIZONE mz) {

	/* Store the mass recycled in each zone in this array */ 
	unsigned int j; 
	double *mass = (double *) malloc ((*mz.mig).n_zones * sizeof(double)); 
	for (j = 0; j < (*mz.mig).n_zones; j++) {
		mass[j] = 0; 
	} 

	/* Look at each tracer particle for continuous recycling */ 
	unsigned long i; 
	for (i = 0l; i < (*mz.mig).tracer_count; i++) {
		TRACER *t = (*mz.mig).tracers[i]; 
		SSP *ssp = mz.zones[(*t).zone_origin] -> ssp; 

		if ((*ssp).continuous) { 
			/* ------------------- Continuous recycling ------------------- */ 
			unsigned long n = (*mz.zones[0]).timestep - (*t).timestep_origin; 
			mass[(*t).zone_current] += (*t).mass * ((*ssp).crf[n + 1l] - 
				(*ssp).crf[n]); 
		} else {} 

	} 

	/* Look at each zone for instantaneous recycling */ 
	for (j = 0; j < (*mz.mig).n_zones; j++) {
		SSP *ssp = mz.zones[j] -> ssp; 

		if (!(*ssp).continuous) {
			/* ------------------ Instantaneous recycling ------------------ */ 
			mass[j] += (
				(*(*mz.zones[j]).ism).star_formation_rate * 
				(*mz.zones[j]).dt * 
				(*(*mz.zones[j]).ssp).R0 
			); 
		} else {} 

	} 

	return mass; 

} 

/* 
 * Evaluate the cumulative return fraction across all timesteps in preparation 
 * of a singlezone simulation. This will store the CRF in the SSP struct 
 * within the singlezone object. 
 * 
 * Parameters 
 * ========== 
 * sz: 		A singlezone object to setup the CRF within 
 * 
 * Returns 
 * ======= 
 * 0 on success, 1 on failure 
 * 
 * header: ssp.h 
 */ 
extern unsigned short setup_CRF(SINGLEZONE *sz) {

	double denominator = CRFdenominator((*(*sz).ssp)); 
	if (denominator < 0) {
		/* 
		 * denominator will be -1 in the case of an unrecognized IMF; return 
		 * 1 on failure 
		 */ 
		return 1; 
	} else { 
		/* 
		 * By design, the singlezone object fills arrays of time-varying 
		 * quantities for ten timesteps beyond the endpoint of the simulation. 
		 * This is a safeguard against memory errors. 
		 */ 
		unsigned long i, n = n_timesteps(*sz); 

		sz -> ssp -> crf = (double *) malloc (n * sizeof(double)); 
		for (i = 0l; i < n; i++) {
			sz -> ssp -> crf[i] = CRFnumerator_Kalirai08(
				(*(*sz).ssp), i * (*sz).dt) / denominator; 
		} 
		return 0; 

	}

}

/* 
 * Determine the cumulative return fraction from a single stellar population 
 * a given time in Gyr after its formation. 
 * 
 * Parameters 
 * ========== 
 * ssp: 		An SSP struct containing information on the stallar IMF and 
 * 				the mass range of star formation 
 * time: 		The age of the stellar population in Gyr 
 * 
 * Returns 
 * ======= 
 * The value of the CRF at that time for the IMF assumptions encoded into the 
 * SSP struct. -1 in the case of an unrecognized IMF 
 * 
 * header: ssp.h 
 */ 
extern double CRF(SSP ssp, double time) { 

	double numerator = CRFnumerator_Kalirai08(ssp, time); 
	if (numerator < 0) {
		/* numerator will be -1 in the case of an unrecognized IMF */ 
		return -1; 
	} else {
		return numerator / CRFdenominator(ssp); 
	}

}

/* 
 * Determine the total mass returned to the ISM from a single stellar 
 * population from all stars a time t in Gyr following their formation. This 
 * is determined by subtracting the Kalirai et al. (2008) model for stellar 
 * remnant masses from the initial mass of stars in this mass range, then 
 * weighting the stellar IMF by this quantity and integrating over the mass 
 * range of star formation. The prefactors are determined in this manner; see 
 * section 2.2 of VICE's science documentation for further details. 
 * 
 * Parameters 
 * ========== 
 * ssp: 		The SSP struct containing information on the stellar IMF and 
 * 				the mass range of star formation 
 * time: 		The time in Gyr following the single stellar population's 
 * 				formation. 
 * 
 * Returns 
 * ======= 
 * The total returned mass in solar masses up to the normalization of the 
 * stellar IMF. -1 in the case of an unrecognized IMF. 
 * 
 * Notes 
 * ===== 
 * This implementation differs mildly from the analytic expression presented 
 * in section 2.2 of VICE's science documentation. This implementation solves 
 * the integral from the turnoff mass to the 8 Msun plus the integral from 
 * 8 Msun to the upper mass limit. 
 * 
 * References 
 * ========== 
 * Kalirai et al. (2008), ApJ, 676, 594 
 * Kroupa (2001), MNRAS, 322, 231 
 */ 
static double CRFnumerator_Kalirai08(SSP ssp, double t) {

	double turnoff_mass = main_sequence_turnoff_mass(t, ssp.postMS); 
	if (turnoff_mass > (*ssp.imf).m_upper) return 0; 
	switch (checksum((*ssp.imf).spec)) {

		case SALPETER: 
			/* Salpeter IMF */ 
			return CRFnumerator_Kalirai08_IMFrange(
				(*ssp.imf).m_upper, 
				turnoff_mass, 
				(*ssp.imf).m_lower, 
				2.35
			); 

		case KROUPA: 
			/* 
			 * Kroupa IMF 
			 * 
			 * Prefactors here come from ensuring continuity at the breaks in 
			 * the power-law indeces of the mass distribution 
			 */ 
			if (turnoff_mass > 0.5) {
				return 0.04 * CRFnumerator_Kalirai08_IMFrange(
					(*ssp.imf).m_upper, 
					turnoff_mass, 
					(*ssp.imf).m_lower, 
					2.3 
				); 
			} else if (0.08 <= turnoff_mass && turnoff_mass <= 0.5) {
				return 0.04 * CRFnumerator_Kalirai08_IMFrange(
					(*ssp.imf).m_upper, 
					turnoff_mass, 
					0.5, 
					2.3
				) + 0.08 * CRFnumerator_Kalirai08_IMFrange(
					0.5, 
					turnoff_mass, 
					(*ssp.imf).m_lower, 
					1.3 
				); 
			} else {
				return 0.04 * CRFnumerator_Kalirai08_IMFrange(
					(*ssp.imf).m_upper, 
					turnoff_mass, 
					0.5, 
					2.3
				) + 0.08 * CRFnumerator_Kalirai08_IMFrange(
					0.5, 
					turnoff_mass, 
					0.08, 
					1.3 
				) + CRFnumerator_Kalirai08_IMFrange(
					0.08, 
					turnoff_mass, 
					(*ssp.imf).m_lower, 
					0.3 
				); 
			}

		case CUSTOM: 
			/* custom IMF -> no assumptions made, must integrate numerically */ 
			ADOPTED_IMF = ssp.imf; 
			INTEGRAL *numerator = integral_initialize(); 
			numerator -> func = &CRFnumerator_integrand; 
			numerator -> a = turnoff_mass; 
			numerator -> b = (*ssp.imf).m_upper; 
			/* default values for these parameters */ 
			numerator -> tolerance = SSP_TOLERANCE; 
			numerator -> method = SSP_METHOD; 
			numerator -> Nmin = SSP_NMIN; 
			numerator -> Nmax = SSP_NMAX; 
			quad(numerator); 
			double x = (*numerator).result; 
			integral_free(numerator); 
			ADOPTED_IMF = NULL; 
			return x; 

		default: 
			/* error handling */ 
			return -1; 

	} 

}

/* 
 * Determine the total mass returned to the ISM from a single stellar 
 * population from a given range of stellar initial mass. This is determined 
 * by subtracting the Kalirai et al. (2008) model for stellar remnant masses 
 * from the initial mass of stars in this mass range, then weighting the 
 * stellar IMF by this quantity and integrating over the mass range of star 
 * formation. The prefactors are determined in this manner; see section 2.2 
 * of VICE's science documentation for further details. 
 * 
 * Parameters 
 * ========== 
 * m_upper: 		The upper mass limit on star formation in Msun 
 * turnoff_mass: 	The main sequence turnoff mass in Msun 
 * m_lower: 		The lower mass limit on star formation in Msun 
 * a: 				The power law index of the stellar IMF. This implementation 
 * 					allows routines that call it to be generalized for 
 * 					piece-wise IMFs like Kroupa (2001). 
 * 
 * Returns 
 * ======= 
 * The total returned mass in solar masses up to the normalization of the 
 * stellar IMF. 
 * 
 * Notes 
 * ===== 
 * This implementation differs mildly from the analytic expression presented 
 * in section 2.2 of VICE's science documentation. This implementation solves 
 * the integral from the turnoff mass to the 8 Msun plus the integral from 
 * 8 Msun to the upper mass limit. 
 * 
 * References 
 * ========== 
 * Kalirai et al. (2008), ApJ, 676, 594 
 * Kroupa (2001), MNRAS, 322, 231 
 */ 
static double CRFnumerator_Kalirai08_IMFrange(double m_upper, 
	double turnoff_mass, double m_lower, double a) { 

	/* 
	 * These functions likely could have been condensed into one method, but 
	 * this is the implementation that seemed to maximize readability 
	 */ 

	if (turnoff_mass < m_lower) { 
		/* 
		 * No more remnants once all stars have died, so report only those 
		 * formed from the relevant range of initial stellar masses. In this 
		 * way, this function can be called with the true turnoff mass, 
		 * letting m_upper and m_lower be the mass bounds on a given 
		 * piece-wise range of the IMF, and the proper value will always be 
		 * returned. 
		 */ 
		return CRFnumerator_Kalirai08_IMFrange(m_upper, m_lower, m_lower, a); 
	} else if (turnoff_mass > m_upper) {
		/* No remnants yet */ 
		return 0; 
	} else if (turnoff_mass >= 8) { 
		/* Stars have died, but only those above 8 Msun */ 
		return CRFnumerator_Kalirai08_above_8Msun(m_upper, turnoff_mass, a); 
	} else { 
		if (m_upper > 8) {
			/* All stars above 8 Msun have died */ 
			return (CRFnumerator_Kalirai08_above_8Msun(m_upper, 8, a) + 
				CRFnumerator_Kalirai08_below_8Msun(8, turnoff_mass, a)); 
		} else {
			/* There never were any stars above 8 Msun to begin with */ 
			return (CRFnumerator_Kalirai08_below_8Msun(m_upper, 
				turnoff_mass, a)); 
		}
			
	}

}

/* 
 * Determine the total mass returned to the ISM from a single stellar 
 * population from stars with initial stellar masses above 8 Msun. This is 
 * determined by subtracting the Kalirai et al. (2008) model for stellar 
 * remnant masses from the initial mass of stars in this mass range, then 
 * weighting the stellar IMF by this quantity and integrating over the mass 
 * range of star formation. The prefactors are determined in this manner; see 
 * section 2.2 of VICE's science documentation for further details. 
 * 
 * Parameters 
 * ========== 
 * m_upper: 			The upper mass limit on star formation in Msun 
 * turnoff_mass: 		The main sequence turnoff mass in Msun 
 * a: 					The power law index of the stellar IMF below 8 Msun. 
 * 
 * Returns 
 * ======= 
 * The returned mass in solar masses up to the normalization of the 
 * stellar IMF 
 * 
 * References 
 * ========== 
 * Kalirai et al. (2008), ApJ, 676, 594 
 * Kroupa (2001), MNRAS, 322, 231 
 */ 
static double CRFnumerator_Kalirai08_above_8Msun(double m_upper, 
	double turnoff_mass, double a) { 

	return (1/(2 - a) * pow(m_upper, 2 - a) - 1.44/(1 - a) * 
		pow(m_upper, 1 - a) - 1/(2 - a) * pow(turnoff_mass, 2 - a) + 
		1.44/(1 - a) * pow(turnoff_mass, 1 - a));

}

/* 
 * Determine the total mass returned to the ISM from a single stellar 
 * population from stars with initial stellar masses below 8 Msun. This is 
 * determined by subtracting the Kalirai et al. (2008) model for stellar 
 * remnant masses from the initial mass of stars in this mass range, then 
 * weighting the stellar IMF by this quantity and integrating over the mass 
 * range of star formation. The prefactors are determined in this manner; see 
 * section 2.2 of VICE's science documentation for further details. 
 * 
 * Parameters 
 * ========== 
 * m_upper: 			The upper mass bound (should always be 8 unless 
 * 						simulating models with no stars above 8 Msun) 
 * turnoff_mass: 		The main sequence turnoff mass in Msun 
 * a: 					The power law index on the stellar IMF below 8 Msun. 
 * 
 * Returns 
 * ======= 
 * The returned mass in solar masses up to the normalization of the 
 * stellar IMF 
 * 
 * References  
 * ========== 
 * Kalirai et al. (2008), ApJ, 676, 594 
 * Kroupa (2001), MNRAS, 322, 231 
 */ 
static double CRFnumerator_Kalirai08_below_8Msun(double m_upper, 
	double turnoff_mass, double a) {

	return (0.891/(2 - a) * pow(m_upper, 2 - a) - 0.394/(1 - a) * 
		pow(m_upper, 1 - a) - 0.891/(2 - a) * pow(turnoff_mass, 2 - a) + 
		0.394/(1 - a) * pow(turnoff_mass, 1 - a));	

} 

/* 
 * Determine the denominator of the cumulative return fraction. This is the 
 * total mass of a single stellar population up to the normalization 
 * constant of the IMF. This is determined by the mass range of star 
 * formation and the IMF itself. See section 2.2 of VICE's science 
 * documentation for details. 
 * 
 * Parameters 
 * ========== 
 * ssp: 		The SSP struct containing information on the IMF and the 
 * 				allowed mass ranges of star formation 
 * 
 * Returns 
 * ======= 
 * The denominator of the cumulative return fraction. When the returned mass 
 * determined by functions in this module is divided by this value, the 
 * CRF is determined. -1 in the case of an unrecognized IMF. 
 */ 
static double CRFdenominator(SSP ssp) {

	switch (checksum((*ssp.imf).spec)) { 

		case SALPETER: 
			/* Salpeter IMF */ 
			return CRFdenominator_IMFrange((*ssp.imf).m_upper, 
				(*ssp.imf).m_lower, 2.35); 

		case KROUPA: 
			/* Kroupa IMF */ 
			if ((*ssp.imf).m_lower > 0.5) {
				return 0.04 * CRFdenominator_IMFrange(
					(*ssp.imf).m_upper, (*ssp.imf).m_lower, 2.3
				); 
			} else if (0.08 <= (*ssp.imf).m_lower && 
				(*ssp.imf).m_lower <= 0.5) {
				return (
					0.04 * CRFdenominator_IMFrange((*ssp.imf).m_upper, 0.5, 2.3) 
					+ 0.08 * CRFdenominator_IMFrange(0.5, (*ssp.imf).m_lower, 
						1.3)
				); 
			} else {
				return (0.04 * CRFdenominator_IMFrange((*ssp.imf).m_upper, 0.5, 
					2.3) + 0.08 * CRFdenominator_IMFrange(0.5, 0.08, 1.3) + 
					CRFdenominator_IMFrange(0.08, (*ssp.imf).m_lower, 0.3) 
				); 
			}

		case CUSTOM: 
			/* custom IMF -> no assumptions made, must integrate numerically */ 
			ADOPTED_IMF = ssp.imf; 
			INTEGRAL *denominator = integral_initialize(); 
			denominator -> func = &CRFdenominator_integrand; 
			denominator -> a = (*ssp.imf).m_lower; 
			denominator -> b = (*ssp.imf).m_upper; 
			/* default values for these properties */ 
			denominator -> tolerance = SSP_TOLERANCE; 
			denominator -> method = SSP_METHOD; 
			denominator -> Nmin = SSP_NMIN; 
			denominator -> Nmax = SSP_NMAX; 
			quad(denominator); 
			double x = (*denominator).result; 
			integral_free(denominator); 
			ADOPTED_IMF = NULL; 
			return x; 

		default: 
			/* error handling */ 
			return -1; 

	} 

}

/* 
 * Determine one term in the denominator of the cumulative return fraction. 
 * See section 2.2 of VICE's science documentation for further details. 
 * 
 * Parameters 
 * ========== 
 * m_upper: 		The upper mass limit on this range of star formation 
 * m_lower: 		The lower mass limit on this range of star formation 
 * a: 				The power law index on the stellar IMF here 
 * 
 * Returns 
 * ======= 
 * The total initial main sequence mass formed in a given range of star 
 * formation, up to the normalization constant of the IMF. 
 */ 
static double CRFdenominator_IMFrange(double m_upper, double m_lower, 
	double a) {

	return 1 / (2 - a) * (pow(m_upper, 2 - a) - pow(m_lower, 2 - a)); 

} 

/* 
 * Evaluate the main sequence mass fraction across all timesteps in preparation 
 * of a singlezone simulation. This will store the MSMF in the SSP struct 
 * within the singlezone object. 
 * 
 * Parameters 
 * ========== 
 * sz: 		A singlezone object to setup the MSMF within 
 * 
 * Returns 
 * ======= 
 * 0 on success, 1 on failure 
 * 
 * header: ssp.h 
 */ 
extern unsigned short setup_MSMF(SINGLEZONE *sz) {

	double denominator = MSMFdenominator((*(*sz).ssp)); 
	if (denominator < 0) {
		/* 
		 * denominator will be -1 in the case of an unrecognized IMF; return 
		 * 1 on failure. 
		 */ 
		return 1; 
	} else {
		/* 
		 * By design, the singlezone object fills arrays of time-varying 
		 * quantities for ten timesteps beyond the endpoint of the simulation. 
		 * This is a safeguard against memory errors. 
		 */ 
		unsigned long i, n = n_timesteps(*sz); 

		sz -> ssp -> msmf = (double *) malloc (n * sizeof(double)); 
		for (i = 0l; i < n; i++) {
			sz -> ssp -> msmf[i] = MSMFnumerator((*(*sz).ssp), 
				i * (*sz).dt) / denominator; 
		} 
		return 0; 
	}

}

/* 
 * Determine the main sequence mass fraction of a stellar population a some 
 * time following its formation. 
 * 
 * Parameters 
 * ========== 
 * ssp: 		A SSP struct containing information on the stellar IMF and 
 * 				the mass range of star formation 
 * time: 		The age of the stellar population in Gyr 
 * 
 * Returns 
 * ======= 
 * The value of the main sequence mass fraction at the specified age. -1 in 
 * the case of an unrecognized IMF. 
 * 
 * header: ssp.h 
 */ 
extern double MSMF(SSP ssp, double time) { 

	double denominator = MSMFdenominator(ssp); 
	if (denominator < 0) { 
		/* MSMFdenominator returns -1 for an unrecognized IMF */ 
		return -1; 
	} else { 
		return MSMFnumerator(ssp, time) / denominator; 
	}

}

/* 
 * Determine the denominator of the main sequence mass fraction. This is 
 * the total initial mass of the main sequence; see section 2.2 of VICE's 
 * science documentation for further details. 
 * 
 * Parameters 
 * ========== 
 * ssp: 		A SSP struct containing information on the stellar IMF and 
 * 				mass range of star formation 
 * 
 * Returns 
 * ======= 
 * The total initial main sequence mass of the stellar population, up to the 
 * normalization constant of the IMF. -1 in the case of an unrecognized IMF. 
 */ 
static double MSMFdenominator(SSP ssp) {

	/* 
	 * The main sequence mass fraction has the same denominator as the 
	 * cumulative return fraction. 
	 */ 
	return CRFdenominator(ssp); 

} 

/* 
 * Determine the numerator of the main sequence mass fraction. This is the 
 * total mass of stars still on the main sequence; see section 2.2 of VICE's 
 * science documentation for further details. 
 * 
 * Parameters
 * ========== 
 * ssp: 		A SSP struct containing information on the stellar IMF and 
 * 				mass range of star formation 
 * time: 		The age of the stellar population in Gyr 
 * 
 * Returns 
 * ======= 
 * The total main sequence mass of the stellar population at the given age, up 
 * to the normalization constant of the IMF. 
 */ 
static double MSMFnumerator(SSP ssp, double t) {

	/* 
	 * The integrated form of the numerator of the main sequence mass fraction 
	 * has the same form as the denominator as the cumulative return fraction, 
	 * but with different bounds. Thus CRFdenominator_IMFrange can be called 
	 * for each of the relevant mass ranges. 
	 */ 

	double turnoff_mass = main_sequence_turnoff_mass(t, ssp.postMS); 

	/* 
	 * First check if it's ouside the mass range of star formation and handle 
	 * appropriately 
	 */ 
	if (turnoff_mass > (*ssp.imf).m_upper) { 
		return MSMFdenominator(ssp); 
	} else if (turnoff_mass < (*ssp.imf).m_lower) {
		return 0; 
	}

	switch(checksum((*ssp.imf).spec)) {

		case SALPETER: 
			/* Salpeter IMF */ 
			return CRFdenominator_IMFrange(turnoff_mass, (*ssp.imf).m_lower, 
				2.35); 

		case KROUPA: 
			/* Kroupa IMF */ 
			if ((*ssp.imf).m_lower < 0.08) {
				/* Need to consider all 3 portions of the Kroupa IMF */ 
				if (turnoff_mass > 0.5) {
					return (
						0.04 * CRFdenominator_IMFrange(turnoff_mass, 0.5, 2.3) + 
						0.08 * CRFdenominator_IMFrange(0.5, 0.08, 1.3) + 
						CRFdenominator_IMFrange(0.08, (*ssp.imf).m_lower, 0.3) 
					); 
				} else if (0.08 <= turnoff_mass && turnoff_mass <= 0.5) {
					return (
						0.08 * CRFdenominator_IMFrange(turnoff_mass, 0.08, 1.3) 
						+ CRFdenominator_IMFrange(0.08, (*ssp.imf).m_lower, 0.3) 
					); 
				} else {
					return CRFdenominator_IMFrange(turnoff_mass, 
						(*ssp.imf).m_lower, 0.3); 
				} 
			} else if (0.08 <= (*ssp.imf).m_lower && (*ssp.imf).m_lower <= 0.5) {
				/* Only two portions of the Kroupa IMF to worry about */ 
				if (turnoff_mass > 0.5) {
					return (
						0.04 * CRFdenominator_IMFrange(turnoff_mass, 0.5, 2.3) + 
						0.08 * CRFdenominator_IMFrange(0.5, (*ssp.imf).m_lower, 
							1.3)
					); 
				} else {
					return 0.08 * CRFdenominator_IMFrange(turnoff_mass, 
						(*ssp.imf).m_lower, 1.3); 
				} 
			} else {
				/* Only the high mass end of the Kroupa IMF to consider */ 
				return 0.04 * CRFdenominator_IMFrange(turnoff_mass, 
					(*ssp.imf).m_lower, 2.3); 
			} 

		case CUSTOM: 
			/* custom IMF -> no assumptions made, must integrate numerically */ 
			ADOPTED_IMF = ssp.imf; 
			INTEGRAL *numerator = integral_initialize(); 
			numerator -> func = &CRFdenominator_integrand; 
			numerator -> a = (*ssp.imf).m_lower; 
			numerator -> b = turnoff_mass; 
			/* default values for these parameters */ 
			numerator -> tolerance = SSP_TOLERANCE; 
			numerator -> method = SSP_METHOD; 
			numerator -> Nmin = SSP_NMIN; 
			numerator -> Nmax = SSP_NMAX; 
			quad(numerator); 
			double x = (*numerator).result; 
			integral_free(numerator); 
			ADOPTED_IMF = NULL; 
			return x; 

		default: 
			/* error handling */ 
			return -1; 

	} 

}

