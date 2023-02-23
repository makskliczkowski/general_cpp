
/**
 * @brief
 *
 * R diagonal has elements smaller than one -> D_m
 * D is already inversed and has elements bigg
 * @param Ql
 * @param Rl
 * @param Pl
 * @param Tl
 * @param Dl
 * @param Qr
 * @param Rr
 * @param Pr
 * @param Tr
 * @param Dr
 * @param Dtmp
 * @return arma::mat
 */
arma::mat inv_left_plus_right_qr(arma::mat& Ql, arma::mat& Rl, arma::umat& Pl, arma::mat& Tl, arma::vec& Dl, arma::mat& Qr, arma::mat& Rr, arma::umat& Pr, arma::mat& Tr, arma::vec& Dr, arma::vec& Dtmp)
{
	const auto loh = true;
	if (loh) {
		// using loh

		makeTwoScalesFromUDT(Rl, Dl);																								// remember D already inversed!
		makeTwoScalesFromUDT(Rr, Dr);																								// remember D already inversed!
		//! D_lm*D_rp^{-1} * X_l * X_r^{-1} + U_l^{-1} * U_r * D_rm * D_lp^{-1}
		setUDTDecomp(
			(DIAG(Rl) * DIAG(Dr)) * Tl * arma::inv(Tr) +
			Ql.t() * Qr * (DIAG(Dl) * DIAG(Rr)),
			Qr, Rl, Pl, Tl, Dtmp);
		//! D_rp^{-1}
		setUDTDecomp(DIAG(Dr) * arma::inv(Qr * DIAG(Rl) * Tl) * DIAG(Dl), Qr, Rl, Pl, Tl, Dtmp);
		//? direct inversion
		//setUDTDecomp(DIAG(Dr) * arma::inv(Qr * DIAG(Rl) * Tl) * DIAG(Dl), Qr, Rl, Pl, Tl);
		return (arma::inv(Tr) * Qr) * DIAG(Rl) * (Tl * Ql.t());
	}
	else {
		setUDTDecomp(
			DIAG(Rl) * Tl * arma::inv(Tr) +
			Ql.t() * Qr * DIAG(Rr),
			Qr, Rl, Pl, Tl, Dtmp);
		return arma::inv(Tl * Tr) * DIAG(Dtmp) * arma::inv(Ql * Qr);
	}
}
