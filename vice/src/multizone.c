/* 
 * This file implements the multizone object and the simulations thereof 
 */ 

#include <stdlib.h> 
#include "multizone.h" 
#include "singlezone.h" 
#include "element.h" 
#include "tracer.h" 
#include "ism.h" 
#include "mdf.h" 
#include "io.h" 

/* ---------- Static function comment headers not duplicated here ---------- */ 
static void multizone_timestepper(MULTIZONE *mz); 
static void multizone_write_history(MULTIZONE mz); 
static void multizone_normalize_MDF(MULTIZONE *mz); 
static void multizone_write_MDF(MULTIZONE mz); 

/* 
 * Allocates memory for and returns a pointer to a multizone object 
 * 
 * Parameters 
 * ========== 
 * n: 		The number of zones in the simulation 
 * 
 * header: multizone.h 
 */ 
extern MULTIZONE *multizone_initialize(int n) {

	MULTIZONE *mz = (MULTIZONE *) malloc (sizeof(MULTIZONE)); 
	mz -> name = (char *) malloc (MAX_FILENAME_SIZE * sizeof(char)); 
	mz -> zones = (SINGLEZONE **) malloc (n * sizeof(SINGLEZONE *)); 
	mz -> migration_matrix_gas = NULL; 
	mz -> migration_matrix_tracers = NULL; 
	mz -> tracers = NULL; 
	mz -> n_zones = n; 

	unsigned int i; 
	for (i = 0; i < n; i++) {
		mz -> zones[i] = singlezone_initialize(); 
	}

	return mz; 

} 

/*
 * Frees the memory stored in a multizone object 
 * 
 * header: multizone.h 
 */ 
extern void multizone_free(MULTIZONE *mz) {

	free(mz -> name); 
	if ((*mz).zones != NULL) {
		unsigned int i; 
		for (i = 0; i < (*mz).n_zones; i++) {
			singlezone_free(mz -> zones[i]); 
		} 
	} else {} 
	if ((*mz).migration_matrix_gas != NULL) free(mz -> migration_matrix_gas); 
	if ((*mz).migration_matrix_tracers != NULL) {
		free(mz -> migration_matrix_tracers); 
	} else {} 
	if ((*mz).tracers != NULL) free(mz -> tracers); 

} 

/* 
 * Runs the multizone simulation under current user settings. 
 * 
 * Parameters 
 * ========== 
 * mz: 		A pointer to the multizone object to run 
 * 
 * Returns
 * ======= 
 * 0 on success, 1 on setup failure 
 * 
 * header: multizone.h 
 */ 
extern int multizone_evolve(MULTIZONE *mz) {

	if (multizone_setup(mz)) return 1; 	/* setup failed */ 

	long n = 0l; 		/* keep track of the number of outputs */ 
	SINGLEZONE *sz = mz -> zones[0]; 		/* for convenience/readability */ 
	while ((*sz).current_time <= (*sz).output_times[(*sz).n_outputs - 1l]) {
		/* 
		 * Run the simulation until the time reaches the final output time 
		 * specified by the user. Write to each zone's history.out file 
		 * whenever an output time is reached, or if the current timestep is 
		 * closer to the next output time than the subsequent timestep. 
		 */ 
		if ((*sz).current_time >= (*sz).output_times[n] || 
			2 * (*sz).output_times[n] < 2 * (*sz).current_time + (*sz).dt) {
			multizone_write_history(*mz); 
			n++; 
		} else {} 
		multizone_timestepper(mz); 
	} 

	/* Normalize all MDFs, write them out, and clean up */ 
	multizone_normalize_MDF(mz); 
	multizone_write_MDF(*mz); 
	multizone_clean(mz); 
	return 0; 

}

/* 
 * Advances all quantities in a multizone object forward one timestep 
 * 
 * Parameters 
 * ========== 
 * mz: 		A pointer to the multizone object to move forward 
 */ 
static void multizone_timestepper(MULTIZONE *mz) {

	unsigned int i, j; 
	update_elements(mz); 
	for (i = 0; i < (*mz).n_zones; i++) { 
		SINGLEZONE *sz = mz -> zones[i]; 
		update_gas_evolution(sz); 

		/* 
		 * Now the ISM in zone i and all of its elements are at the next 
		 * timestep. Bookkeep the new metallicity 
		 */ 
		for (j = 0; j < (*sz).n_elements; j++) {
			sz -> elements[j] -> Z[(*sz).timestep + 1l] = (
				(*(*sz).elements[j]).mass / (*(*sz).ism).mass 
			); 
		} 
		update_MDF(sz); 

		sz -> current_time += (*sz).dt; 
		sz -> timestep++; 
	}

} 

/*
 * Sets up every zone in a multizone object for simulation 
 * 
 * Parameters 
 * ========== 
 * mz: 		A pointer to the multizone object itself 
 * 
 * Returns 
 * ======= 
 * 0 on success, 1 on failure 
 * 
 * header: multizone.h 
 */ 
extern int multizone_setup(MULTIZONE *mz) { 

	unsigned int i; 
	for (i = 0; i < (*mz).n_zones; i++) {
		if (singlezone_setup(mz -> zones[i])) {
			return 1; 
		} else { 
			continue; 
		} 
	} 
	return 0; 

} 

/* 
 * Frees up the memory allocated in running a multizone simulation. This does 
 * not free up the memory stored by simplying having a multizone object in the 
 * python interpreter. That is cleared by calling multizone_free. 
 * 
 * Parameters 
 * ========== 
 * mz: 		A pointer to the multizone object to clean 
 * 
 * header: multizone.h 
 */ 
extern void multizone_clean(MULTIZONE *mz) {

	/* clean each singlezone object */ 
	unsigned int i; 
	for (i = 0; i < (*mz).n_zones; i++) { 
		singlezone_close_files(mz -> zones[i]); 
		singlezone_clean(mz -> zones[i]); 
	} 

	/* free up each tracer and set the pointer to NULL again */ 
	long j; 
	for (j = 0l; 
		j < (*(*mz).zones[0]).timestep * (*mz).n_zones * (*mz).n_tracers; 
		j++) {
		tracer_free(mz -> tracers[j]); 
	} 
	free(mz -> tracers); 
	mz -> tracers = NULL; 

	/* free up the migration matrices */ 
	free(mz -> migration_matrix_gas); 
	free(mz -> migration_matrix_tracers); 
	mz -> migration_matrix_gas = NULL; 
	mz -> migration_matrix_tracers = NULL; 

} 

/* 
 * Writes history output for each zone in a multizone simulation 
 * 
 * Parameters 
 * ========== 
 * mz: 		The multizone object to write output from 
 */ 
static void multizone_write_history(MULTIZONE mz) { 

	unsigned int i; 
	for (i = 0; i < mz.n_zones; i++) {
		write_history_output(*mz.zones[i]); 
	} 

} 

/* 
 * Normalizes the stellar MDFs in all zones in a multizone object. 
 * 
 * Parameters 
 * ========== 
 * mz: 		A pointer to the multizone object for the current simulation 
 */ 
static void multizone_normalize_MDF(MULTIZONE *mz) {

	unsigned int i; 
	for (i = 0; i < (*mz).n_zones; i++) {
		normalize_MDF(mz -> zones[i]); 
	} 

} 

/* 
 * Writes the stellar MDFs to all output files. 
 * 
 * Parameters 
 * ========== 
 * mz: 		The multizone object to write the MDF from 
 */ 
static void multizone_write_MDF(MULTIZONE mz) {

	unsigned int i; 
	for (i = 0; i < mz.n_zones; i++) {
		write_mdf_output(*mz.zones[i]); 
	}

}



