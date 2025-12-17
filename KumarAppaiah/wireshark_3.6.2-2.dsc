-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA512

Format: 3.0 (quilt)
Source: wireshark
Binary: wireshark-common, wireshark, wireshark-qt, wireshark-gtk, tshark, wireshark-dev, wireshark-doc, libwireshark15, libwsutil13, libwsutil-dev, libwireshark-data, libwireshark-dev, libwiretap12, libwiretap-dev
Architecture: any all
Version: 3.6.2-2
Maintainer: Balint Reczey <balint@balintreczey.hu>
Homepage: https://www.wireshark.org/
Standards-Version: 4.6.0.1
Vcs-Browser: https://salsa.debian.org/debian/wireshark
Vcs-Git: https://salsa.debian.org/debian/wireshark.git -b debian/master
Testsuite: autopkgtest
Testsuite-Triggers: at-spi2-core, xauth, xvfb
Build-Depends: qtbase5-dev, qtbase5-dev-tools, qttools5-dev, qttools5-dev-tools, qtmultimedia5-dev, libpcap0.8-dev, flex, libz-dev, debhelper (>= 12), po-debconf, python3, python3-ply, libc-ares-dev, xsltproc, dh-python, docbook-xsl, docbook-xml, libxml2-utils, libcap2-dev | libcap-dev, lsb-release, libparse-yapp-perl, libglib2.0-dev, libgnutls28-dev, libgcrypt-dev, libkrb5-dev, liblua5.2-dev, libsmi2-dev, libmaxminddb-dev, libsystemd-dev, libnl-genl-3-dev [linux-any], libnl-route-3-dev [linux-any], asciidoctor, cmake (>= 3.5), libsbc-dev, libnghttp2-dev, libssh-gcrypt-dev, liblz4-dev, libsnappy-dev, libzstd-dev, libspandsp-dev, libxml2-dev, libbrotli-dev, libspeexdsp-dev, libminizip-dev, libbcg729-dev
Build-Conflicts: libsnmp-dev, libsnmp4.2-dev
Package-List:
 libwireshark-data deb libs optional arch=all
 libwireshark-dev deb libdevel optional arch=any
 libwireshark15 deb libs optional arch=any
 libwiretap-dev deb libdevel optional arch=any
 libwiretap12 deb libs optional arch=any
 libwsutil-dev deb libdevel optional arch=any
 libwsutil13 deb libs optional arch=any
 tshark deb net optional arch=any
 wireshark deb net optional arch=any
 wireshark-common deb net optional arch=any
 wireshark-dev deb devel optional arch=any
 wireshark-doc deb doc optional arch=all
 wireshark-gtk deb net optional arch=any
 wireshark-qt deb net optional arch=any
Checksums-Sha1:
 ba5f38a333558593321e485fc3dcd280377b2592 39617868 wireshark_3.6.2.orig.tar.xz
 a27228a897c1c872b2eed04243039cd718cd7487 76408 wireshark_3.6.2-2.debian.tar.xz
Checksums-Sha256:
 05dd94fcd6e41624828915e4b61458cb8255bf4c801fcc52b48455848933c2b1 39617868 wireshark_3.6.2.orig.tar.xz
 78fd18f5fed2c82dac4910c07bc87838e9fcf1fcf945fb73fa35ac38f1720057 76408 wireshark_3.6.2-2.debian.tar.xz
Files:
 d38da2b948e30da5c579f82880120216 39617868 wireshark_3.6.2.orig.tar.xz
 d9d95e66054ede5000aff3d3dfc68f7f 76408 wireshark_3.6.2-2.debian.tar.xz

-----BEGIN PGP SIGNATURE-----

iQIzBAEBCgAdFiEEI/PvTgXX55rLQUfDg6KBkWslS0UFAmIiSQUACgkQg6KBkWsl
S0WYDRAAlcFgEOhbrdbsHWwOGBxXQM51KzIiVOGaDIwP9UZI0AEGLfULIcMw3KDA
U8tvQA4Urd2fGekfl0yC8CeJ+OXCqx3legEogBLEwAeQNIsN/EYUFOR04fmIMZY3
DRK+MROmQ+MDBOKV9UzLu+9gdJ8uCnK0PIjYux14GE8ymv60FKPeWYS7mZ1mWjAm
5Q4dR6ntiRDJUmQhpgf6qedwAuZetjXNZk+bm45GT5+tK/bJolozA8qlgQGEuSHH
+YgU5A/5Y1oT6hTsKyN3I9LH6BwZiM2HA/QE7+IotxQcaLZ1nYcwzmT1N5++XKFY
WsezKthFzhZQ6Fo9nj0o/z0g/r6gbkiGs4VugGt+rZx6JjUMb3Aec2HkUoF/4qzN
njYMDJ4E15OEhdfqiEmEbFqhn3fZItC3vqZTPBZ//ChP6OPEXgZM/8LfbdXNlF6o
hnA1IXW9CgAsF9HdtW8TCdimjqVgf9K+eQoqn4TCwaGMJZUWbedyU2GJPD4eMcz8
hWzZq3FSCrzF8cVg+WV4ub2RpDpVXVA2VXi1+Zq8J7mGsAmkEPku82/uiZlB7XLn
zyOKSb61OwHO56n0OQ8mwlLz1CctixguxW51BJmvnpeR9L31JHopWw8mO+FY4+X/
DnGAQz34qJE7KPUTjdFGlNl9SyIE/O9zDFzbJA+pua6KsggSpKM=
=W0vz
-----END PGP SIGNATURE-----
