auto isZero = [&](const Vector2D &vec, float tolerance = 0.05)->bool {
		return std::abs(vec.x) < tolerance && std::abs(vec.y) < tolerance;
	};

	auto tripleProduct = [&](const Vector2D &v1, const Vector2D &v2, const Vector2D &v3)->Vector2D {
		double p = v1.x * v2.y - v1.y * v2.x;
		return Vector2D(-p * v3.y, p * v3.x);
	};

	auto oppositeNegate = [&](const Vector2D &vec)->Vector2D {
		return Vector2D(-vec.x, -vec.y);
	};

	auto getFarthestPoint = [&](const Vector2D &dir, const Vector2D poly[], int len)->Vector2D {
		Vector2D bestPoint = poly[0];
		float bestProj = bestPoint.dotProduct(dir);
		for (int i = 1; i < len; ++i) {
			Vector2D curr = poly[i];
			double proj = curr.dotProduct(dir);
			if (proj > bestProj) {
				bestPoint = curr;
				bestProj = proj;
			}
		}
		return bestPoint;
	};

	auto getSupportPoint = [&](const Vector2D &dir, const Vector2D poly1[], int len1, const Vector2D poly2[], int len2)->Vector2D {
		Vector2D v1 = getFarthestPoint(dir, poly1, len1),
		v2 = getFarthestPoint(oppositeNegate(dir), poly2, len2);
		return v1 - v2;
	};

	auto isIntersect = [&](const Vector2D poly1[], int len1, const Vector2D poly2[], int len2)->bool {
		Vector2D simplexA, simplexB, simplexC, dir = Vector2D(-1, -1);

		simplexA = getSupportPoint(dir, poly1, len1, poly2, len2);
		if (simplexA.dotProduct(dir) <= 0) {
			return false;
		}

		dir = oppositeNegate(simplexA);
		simplexB = getSupportPoint(dir, poly1, len1, poly2, len2);
		if (simplexB.dotProduct(dir) <= 0) {
			return false;
		}

		Vector2D ab = simplexB - simplexA;
		dir = tripleProduct(ab, oppositeNegate(simplexA), ab);

		for (int i = 25; i--;) {
			if (isZero(dir)) {
				return true;
			}

			simplexC = getSupportPoint(dir, poly1, len1, poly2, len2);
			if (simplexC.dotProduct(dir) <= 0) {
				return false;
			}

			Vector2D ba = simplexA - simplexB;
			Vector2D ac = simplexC - simplexA;
			Vector2D bc = simplexC - simplexB;
			Vector2D acPerp = tripleProduct(ac, oppositeNegate(ba), ac);
			Vector2D bcPerp = tripleProduct(bc, ba, bc);

			if (acPerp.dotProduct(simplexA) > 0) {
				simplexB = simplexC;
				dir = oppositeNegate(acPerp);
			}
			else if (bcPerp.dotProduct(simplexB) > 0) {
				simplexA = simplexC;
				dir = oppositeNegate(bcPerp);
			}
			else {
				return true;
			}
		}

		return false;
	};